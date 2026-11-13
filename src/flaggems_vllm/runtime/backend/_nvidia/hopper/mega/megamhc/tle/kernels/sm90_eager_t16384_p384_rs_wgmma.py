"""Production exact-``T=16384`` facade for the shared P384 eager kernel.

Uses the fixed ``H=4096, R=4, P=24`` manual512 configuration: 12 producer +
4 RS-WGMMA warps, four K64 stages, TMA Fn staging, route12 tail, maxnreg 64.
"""

from __future__ import annotations

from typing import Mapping, cast

from . import sm90_eager_p384_rs_wgmma as _base

TOKENS = 16384
SHAPE_CONFIG = _base.P384ShapeConfig(TOKENS)
NUM_M_TILES = SHAPE_CONFIG.num_m_tiles
GRID_CTAS = NUM_M_TILES
NUM_WARPS = _base.NUM_WARPS
NUM_K64_BLOCKS = _base.NUM_K64_BLOCKS
ACTIVE_K64_BLOCKS = _base.ACTIVE_K64_BLOCKS
TAIL_BATCHES = _base.TAIL_BATCHES
TAIL_WORK_PHASE = _base.TAIL_WORK_PHASE

_sm90_eager_t16384_p384_rs_wgmma_kernel = _base._sm90_eager_p384_rs_wgmma_kernel


class EagerT16384P384Plan(_base.P384Plan):
    """Exact-shape Plan type retained for the admitted launch adapter."""


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> EagerT16384P384Plan:
    return cast(
        EagerT16384P384Plan,
        _base.prepare_p384_plan(
            inputs,
            outputs,
            shape_config=SHAPE_CONFIG,
            plan_type=EagerT16384P384Plan,
        ),
    )


__all__ = [
    "EagerT16384P384Plan",
    "GRID_CTAS",
    "NUM_M_TILES",
    "TOKENS",
    "prepare_tle_sm90_mega_mhc",
]
