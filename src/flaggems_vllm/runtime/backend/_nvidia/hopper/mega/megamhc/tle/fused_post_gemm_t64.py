"""Exact-T64 compatibility facade for the shared SM90 M64 kernel family.

Historically T=64 and padded T=16..56 carried byte-near-identical 4k-line
device implementations.  The canonical body now lives in
``kernels.sm90_padded_m64_wgmma``.  This module retains the established T64
module, Plan type, kernel symbol, and legacy environment contract consumed by
the prepared-launch registry; shape selection remains setup-only and the hot
path still launches the same single specialized kernel directly.
"""

from __future__ import annotations

import os
from dataclasses import fields
from typing import Mapping

_LOGICAL_TOKENS_ENV = "DG_TLE_MHC_PADDED_WGMMA_LOGICAL_TOKENS"
_logical_tokens = os.environ.get(_LOGICAL_TOKENS_ENV)
if _logical_tokens not in (None, "64"):
    raise RuntimeError(
        f"{_LOGICAL_TOKENS_ENV}={_logical_tokens!r}, but the T64 facade "
        "requires '64'"
    )
os.environ[_LOGICAL_TOKENS_ENV] = "64"

# Preserve the old public tuning contract without duplicating the device
# implementation.  Dispatch still sets the T64 names; the canonical module
# consumes the corresponding padded-family names before it is imported.
_LEGACY_ENV_ALIASES = {
    "DG_TLE_MHC_T32_WGMMA_BM64_DOT": "DG_TLE_MHC_T64_BM64_DOT",
    "DG_TLE_MHC_T32_WGMMA_NUM_SPLITS": "DG_TLE_MHC_T64_NUM_SPLITS",
    "DG_TLE_MHC_T32_WGMMA_NUM_WARPS": "DG_TLE_MHC_T64_NUM_WARPS",
    "DG_TLE_MHC_T32_WGMMA_STAGED_MATH_REGS": ("DG_TLE_MHC_T64_STAGED_MATH_REGS"),
    "DG_TLE_MHC_T32_WGMMA_EXPLICIT_WGMMA": ("DG_TLE_MHC_T64_EXPLICIT_WGMMA"),
    "DG_TLE_MHC_T32_WGMMA_STAGED_WGMMA": "DG_TLE_MHC_T64_STAGED_WGMMA",
    "DG_TLE_MHC_T32_WGMMA_WARP_OVERLAP": "DG_TLE_MHC_T64_WARP_OVERLAP",
    "DG_TLE_MHC_T32_WGMMA_DIRECT_ROUTING_TAIL": ("DG_TLE_MHC_T64_DIRECT_ROUTING_TAIL"),
    "DG_TLE_MHC_T32_WGMMA_DIRECT_TAIL_ILP4": ("DG_TLE_MHC_T64_DIRECT_TAIL_ILP4"),
    "DG_TLE_MHC_T32_WGMMA_PERSISTENT_WS": "DG_TLE_MHC_T64_PERSISTENT_WS",
    "DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4": ("DG_TLE_MHC_T64_PERSISTENT_ROUTE4"),
    "DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4_REDUCTION": (
        "DG_TLE_MHC_T64_PERSISTENT_ROUTE4_REDUCTION"
    ),
    "DG_TLE_MHC_T32_WGMMA_ROUTE4_WARP0_FINALIZE": (
        "DG_TLE_MHC_T64_ROUTE4_WARP0_FINALIZE"
    ),
    "DG_TLE_MHC_T32_WGMMA_SPLIT128_PAIR_ROUTE": ("DG_TLE_MHC_T64_SPLIT128_PAIR_ROUTE"),
    "DG_TLE_MHC_T32_WGMMA_EARLY_PRE_SYNC": ("DG_TLE_MHC_T64_EARLY_PRE_SYNC"),
    "DG_TLE_MHC_T32_WGMMA_PHASE_PROFILE": "DG_TLE_MHC_T64_PHASE_PROFILE",
}

for _canonical_name, _legacy_name in _LEGACY_ENV_ALIASES.items():
    _value = os.environ.get(_legacy_name)
    if _value is None and _legacy_name == "DG_TLE_MHC_T64_PHASE_PROFILE":
        _value = os.environ.get("DG_TLE_MHC_PHASE_PROFILE")
    if _value is None:
        continue
    _current = os.environ.get(_canonical_name)
    if _current not in (None, _value):
        raise RuntimeError(
            f"{_canonical_name}={_current!r}, but {_legacy_name} requires "
            f"{_value!r}"
        )
    os.environ[_canonical_name] = _value


from .kernels import sm90_padded_m64_wgmma as _base  # noqa: E402

if _base.LOGICAL_TOKENS != 64:
    raise RuntimeError(
        "the shared M64 kernel was already specialized for "
        f"T={_base.LOGICAL_TOKENS}; construct token buckets through the "
        "isolated setup path"
    )

TOKENS = _base.TOKENS
GRID_CTAS = _base.GRID_CTAS
NUM_WARPS = _base.NUM_WARPS

# The explicit registry resolves this historical symbol from this module.
_t64_bm64_split64_tf32_kernel = _base._t32_wgmma_bm64_split128_tf32_kernel

# Explicit helper aliases keep Triton/FlagTree global-dependency discovery on
# the same JITFunction objects as before.  ``__getattr__`` below is only a
# compatibility fallback for non-kernel diagnostics.
_physical_thread_id_x = _base._physical_thread_id_x
_monotonic_grid128_arrive_and_wait = _base._monotonic_grid128_arrive_and_wait
_direct_hidden_tail = _base._direct_hidden_tail
_direct_routing_finalize = _base._direct_routing_finalize
_fast_rcp_ftz_f32 = _base._fast_rcp_ftz_f32
_warp_broadcast_lane24_f32 = _base._warp_broadcast_lane24_f32
_warp_reduce4_max = _base._warp_reduce4_max
_warp_reduce4_sum = _base._warp_reduce4_sum
_warp_reduce4x4_column_sum = _base._warp_reduce4x4_column_sum


class T64BM64DotPlan(_base.PaddedM64WGMMAPlan):
    """Exact-T64 public Plan type retained for launch-registry compatibility."""


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> T64BM64DotPlan:
    prepared = _base.prepare_tle_sm90_mega_mhc(inputs, outputs)
    metadata = dict(prepared.metadata)
    result_scope = str(metadata["result_scope"]).replace("small_padded", "t64", 1)
    metadata.update(
        {
            "T": 64,
            "padded_T": 64,
            "logical_tokens": 64,
            "kernel_symbol": "_t64_bm64_split_tf32_kernel",
            "result_scope": result_scope,
        }
    )
    metadata.pop("invalid_m_row_policy", None)
    values = {
        field.name: (
            metadata if field.name == "metadata" else getattr(prepared, field.name)
        )
        for field in fields(_base.PaddedM64WGMMAPlan)
    }
    return T64BM64DotPlan(**values)


# Other admitted families historically import reduction/tail helpers from this
# module.  Forward those attributes without copying their implementations.
def __getattr__(name: str):
    return getattr(_base, name)


__all__ = ["T64BM64DotPlan", "prepare_tle_sm90_mega_mhc"]
