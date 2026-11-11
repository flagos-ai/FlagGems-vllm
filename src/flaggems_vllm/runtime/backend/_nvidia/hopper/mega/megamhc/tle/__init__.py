"""Hopper TLE implementations for MegaMHC.

The stable kernel families are imported lazily.  Several research kernels
specialize their geometry from environment variables at module import time;
eagerly importing one family here would therefore make an unrelated family's
configuration depend on import order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .fused_post_gemm_t64 import T64BM64DotPlan

__all__ = [
    "EAGER_TOKENS",
    "GRAPH_TOKENS",
    "MissingTLECaseError",
    "T64BM64DotPlan",
    "candidate_family",
    "prepare_tle_sm90_mega_mhc",
]


def __getattr__(name: str) -> Any:
    """Load shape dispatch lazily without compiling unrelated kernels."""
    if name == "T64BM64DotPlan":
        from . import fused_post_gemm_t64

        return getattr(fused_post_gemm_t64, name)
    if name in __all__:
        from . import dispatch

        return getattr(dispatch, name)
    raise AttributeError(name)
