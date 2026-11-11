"""Production admission wrapper for the exact T=448 split16 RS kernel.

The underlying TLE implementation is shared with the independently admitted
T=432 specialization.  This module owns the complete T=448 import-time
configuration and pins the implementation source hash, so production dispatch
does not depend on ambient research flags and evidence for one token bucket
cannot silently admit another.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Mapping


TOKENS = 448
_EXPECTED_IMPLEMENTATION_SHA256 = (
    "9aca5026b6bfdd29f54278a9971c06ddb26a8b49b01082e7433bc39a6c5460cc"
)
_EXACT_ENVIRONMENT = {
    "DG_TLE_MHC_LARGE_GRAPH_ROLE4": "1",
    "DG_TLE_MHC_LARGE_GRAPH_ROLE4_LOGICAL_TOKENS": str(TOKENS),
    "DG_TLE_MHC_LARGE_GRAPH_TRUE_K64_STAGE": "1",
    "DG_TLE_MHC_LARGE_GRAPH_COEFF_CACHE": "0",
    "DG_TLE_MHC_LARGE_GRAPH_WEIGHT_CACHE": "0",
    "DG_TLE_MHC_LARGE_GRAPH_PTX_SINKHORN": "1",
    "DG_TLE_MHC_LARGE_GRAPH_NON_STRIPED_TAIL": "0",
    "DG_TLE_MHC_LARGE_GRAPH_MATH_REGS": "128",
    "DG_TLE_MHC_LARGE_GRAPH_EARLY_PRE_SYNC": "1",
    "DG_TLE_MHC_LARGE_GRAPH_PROBE_PHASE": "4",
    "DG_TLE_MHC_LARGE_GRAPH_ROLE_OVERLAP": "0",
    "DG_TLE_MHC_LARGE_GRAPH_ALLOCATED_ROLE_BARRIERS": "0",
    "DG_TLE_MHC_LARGE_GRAPH_SHARED_ROLE_FLAGS": "0",
    "DG_TLE_MHC_LARGE_GRAPH_MANUAL_ROLE4_TAIL": "0",
    "DG_TLE_MHC_LARGE_GRAPH_PTX_BF16X8_TAIL": "0",
    "DG_TLE_MHC_LARGE_GRAPH_FAST_GRID_BARRIER": "0",
    "DG_TLE_MHC_LARGE_GRAPH_TILE_LOCAL_BARRIER": "0",
    "DG_TLE_MHC_LARGE_GRAPH_EXACT_T448": "0",
    "DG_TLE_MHC_LARGE_GRAPH_EXACT_T512": "0",
    "DG_TLE_MHC_LARGE_GRAPH_SPLIT16_RS_MATH_REGS": "128",
}


def _set_exact_environment() -> None:
    for name, expected in _EXACT_ENVIRONMENT.items():
        actual = os.environ.get(name)
        if actual not in (None, expected):
            raise RuntimeError(
                f"{name}={actual!r}, but exact T448 requires {expected!r}"
            )
        os.environ[name] = expected


_set_exact_environment()

from . import sm90_large_graph_split16_rs_wgmma as _implementation  # noqa: E402


_IMPLEMENTATION_SOURCE = Path(_implementation.__file__).resolve()
_IMPLEMENTATION_SHA256 = hashlib.sha256(
    _IMPLEMENTATION_SOURCE.read_bytes()
).hexdigest()
if _IMPLEMENTATION_SHA256 != _EXPECTED_IMPLEMENTATION_SHA256:
    raise RuntimeError(
        "T448 admission pins split16 RS implementation SHA256 "
        f"{_EXPECTED_IMPLEMENTATION_SHA256}, got {_IMPLEMENTATION_SHA256}"
    )
if _implementation.TOKENS != TOKENS:
    raise RuntimeError(
        f"T448 admission imported T={_implementation.TOKENS} implementation"
    )


@dataclass(frozen=True)
class LargeGraphSplit16RST448Plan:
    """Shape-exact production plan delegating only the audited kernel body."""

    implementation: _implementation.LargeGraphSplit16RSPlan
    metadata: dict[str, object]

    @property
    def grid_ctas(self) -> int:
        return self.implementation.grid_ctas

    @property
    def partials(self):
        return self.implementation.partials

    @property
    def workspace_tensors(self):
        return self.implementation.workspace_tensors

    @property
    def counter_tensors(self):
        return self.implementation.counter_tensors

    def compile(self):
        return self.implementation.compile()

    def run(self) -> None:
        self.implementation.run()


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> LargeGraphSplit16RST448Plan:
    implementation = _implementation.prepare_tle_sm90_mega_mhc(inputs, outputs)
    metadata = dict(implementation.metadata)
    metadata.update({
        "admission_shape": "T448",
        "admission_source": str(Path(__file__).resolve()),
        "implementation_source": str(_IMPLEMENTATION_SOURCE),
        "implementation_source_sha256": _IMPLEMENTATION_SHA256,
        "audited_spill_admission": True,
        "expected_spills": 2,
        "expected_local_bytes_per_thread": 8,
        "experimental": False,
        "dispatch_enabled": True,
    })
    return LargeGraphSplit16RST448Plan(
        implementation=implementation,
        metadata=metadata,
    )


__all__ = [
    "LargeGraphSplit16RST448Plan",
    "TOKENS",
    "prepare_tle_sm90_mega_mhc",
]
