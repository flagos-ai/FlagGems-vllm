"""Production admission wrapper for the exact T=448 split16 RS kernel.

The underlying TLE implementation is shared with T=432.  Production dispatch
selects this module only for ``T=448, H=4096, R=4, P=24`` and fixes seven M64
tiles, split-16, 112 producer CTAs, sixteen producer warps, four RS-WGMMA
warps, eight stages, 128 math registers, and striped token ownership.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

TOKENS = 448
_EXACT_ENVIRONMENT = {
    "DG_TLE_MHC_LARGE_GRAPH_ROLE4_LOGICAL_TOKENS": str(TOKENS),
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
    metadata.update(
        {
            "admission_shape": "T448",
            "audited_spill_admission": True,
            "expected_spills": 2,
            "expected_local_bytes_per_thread": 8,
            "experimental": False,
            "dispatch_enabled": True,
        }
    )
    return LargeGraphSplit16RST448Plan(
        implementation=implementation,
        metadata=metadata,
    )


__all__ = [
    "LargeGraphSplit16RST448Plan",
    "TOKENS",
    "prepare_tle_sm90_mega_mhc",
]
