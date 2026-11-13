"""Production exact-T496 facade for the shared split16 RS all-producer kernel.

The selected configuration is H4096/R4/P24, eight M64 tiles, split-16, 128
producer CTAs, sixteen producer warps, four RS-WGMMA warps, and 128 math
registers.  No device body is duplicated here.
"""

from __future__ import annotations

import os
from typing import Mapping

TOKENS = 496
_TOKEN_ENV = "DG_TLE_MHC_LARGE_GRAPH_ROLE4_LOGICAL_TOKENS"
_current = os.environ.get(_TOKEN_ENV)
if _current not in (None, str(TOKENS)):
    raise RuntimeError(f"{_TOKEN_ENV}={_current!r}, but the T496 facade requires '496'")
os.environ[_TOKEN_ENV] = str(TOKENS)


from . import sm90_large_graph_split16_rs_t464_nonstriped_wgmma as _shared  # noqa: E402

if _shared.TOKENS != TOKENS:
    raise RuntimeError(
        f"shared exact split16 module is specialized for T={_shared.TOKENS}, "
        f"expected T={TOKENS}"
    )
_implementation = _shared._implementation
_t496_allproducer_kernel = _shared._t464_allproducer_kernel


class LargeGraphSplit16RST496Plan(_shared.LargeGraphSplit16RST464Plan):
    """Public T496 Plan type retained for launch-registry compatibility."""


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> LargeGraphSplit16RST496Plan:
    prepared = _shared.prepare_tle_sm90_mega_mhc(inputs, outputs)
    metadata = dict(prepared.metadata)
    metadata.update(
        {
            "admission_shape": "T496",
            "kernel_symbol": "_t496_allproducer_kernel",
            "pipeline_variant": ("t496-allproducer-nonstriped-active-cta-skip-tilebar"),
        }
    )
    return LargeGraphSplit16RST496Plan(
        implementation=prepared.implementation,
        metadata=metadata,
    )


__all__ = [
    "LargeGraphSplit16RST496Plan",
    "TOKENS",
    "prepare_tle_sm90_mega_mhc",
]
