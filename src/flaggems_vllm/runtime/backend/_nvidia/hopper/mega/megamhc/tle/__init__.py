"""Hopper TLE implementations for MegaMHC.

The production kernel families are imported lazily because their exact token
geometry is specialized at module import time. Eagerly importing one family
here would make an unrelated family's configuration depend on import order.
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
