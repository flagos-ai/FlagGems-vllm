"""Production eager MegaMHC facade for exact ``T=4100``.

The fixed Normal-BF16 profile is ``H=4096, R=4, P=24``.  Four resident CTAs
split H4096 into H1024 ranges and synchronize through the admitted global tile
barrier; the final M64 tile contains four live rows.  The shared P256 base uses
12 producer and 4 RS-WGMMA warps, phase-3 x8 tails, cp.async Fn staging, and
``maxnreg=64``.  The complete operator remains one kernel launch.
"""

from __future__ import annotations

from typing import Mapping

from . import sm90_eager_t3076_p256_cluster16_manual512_cpasync as _base

TOKENS = 4100
NUM_M_TILES = (TOKENS + _base.BLOCK_M - 1) // _base.BLOCK_M
CLUSTER_CTAS = 4
NUM_SPLITS = CLUSTER_CTAS
SPLIT_HIDDEN = _base.HIDDEN // NUM_SPLITS
MAX_SPLITS = NUM_SPLITS
GRID_CTAS = NUM_M_TILES * NUM_SPLITS

_base.TOKENS = TOKENS
_base.NUM_M_TILES = NUM_M_TILES
_base.CLUSTER_CTAS = CLUSTER_CTAS
_base.NUM_SPLITS = NUM_SPLITS
_base.SPLIT_HIDDEN = SPLIT_HIDDEN
_base.MAX_SPLITS = MAX_SPLITS
_base.GRID_CTAS = GRID_CTAS
_base.JIT_TOKENS = _base.tl.constexpr(TOKENS)
_base.JIT_NUM_M_TILES = _base.tl.constexpr(NUM_M_TILES)
_base.JIT_NUM_SPLITS = _base.tl.constexpr(NUM_SPLITS)
_base.JIT_SPLIT_HIDDEN = _base.tl.constexpr(SPLIT_HIDDEN)
_base.JIT_MAX_SPLITS = _base.tl.constexpr(MAX_SPLITS)

EagerT4100P1024Cluster4Manual512ProjectionPlan = (
    _base.EagerT3076P256Cluster16Manual512ProjectionPlan
)


def prepare_tle_sm90_projection(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> EagerT4100P1024Cluster4Manual512ProjectionPlan:
    """Prepare the T=4100 candidate with its real four-token tail tile."""
    plan = _base.prepare_tle_sm90_projection(inputs, outputs)
    plan.metadata.update(
        {
            "T": TOKENS,
            "grid_ctas": GRID_CTAS,
            "num_m_tiles": NUM_M_TILES,
            "logical_grid_tiles": NUM_M_TILES,
            "cluster_ctas": CLUSTER_CTAS,
            "num_splits": NUM_SPLITS,
            "split_hidden": SPLIT_HIDDEN,
            "active_k64_blocks": SPLIT_HIDDEN // 64,
            "cta_mapping": "tile=pid//4, split=pid%4",
            "shape_specialization": "T=4100 H1024 cluster4 partial-M4",
        }
    )
    return plan


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> EagerT4100P1024Cluster4Manual512ProjectionPlan:
    """Shape-exact full-operator seam; still exactly one TLE launch."""
    plan = prepare_tle_sm90_projection(inputs, outputs)
    if not plan.tail_enabled:
        raise RuntimeError("the T=4100 cluster4 experiment requires full tail")
    return plan


__all__ = [
    "CLUSTER_CTAS",
    "GRID_CTAS",
    "MAX_SPLITS",
    "NUM_M_TILES",
    "NUM_SPLITS",
    "SPLIT_HIDDEN",
    "TOKENS",
    "EagerT4100P1024Cluster4Manual512ProjectionPlan",
    "prepare_tle_sm90_mega_mhc",
    "prepare_tle_sm90_projection",
]
