"""Production eager MegaMHC facade for exact ``T=2048``.

The fixed Normal-BF16 profile is ``H=4096, R=4, P=24``. Thirty-two logical
M64 token tiles each expand to a sixteen-CTA Hopper cluster, with one H256
projection split per CTA. Projection, routing, and the BF16-boundary RMSNorm
tail execute in one kernel.

Cluster size sixteen is admitted from the compiled kernel and active device
before first launch. Devices or configurations whose maximum cluster size is
smaller fail closed; this path never relies on a normal-launch software spin
barrier.
"""

from __future__ import annotations

from typing import Mapping

from . import sm90_eager_t3076_p256_cluster16_manual512_cpasync as _base

TOKENS = 2048
NUM_M_TILES = TOKENS // _base.BLOCK_M
CLUSTER_CTAS = 16
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

EagerT2048P256Manual512ProjectionPlan = (
    _base.EagerT3076P256Cluster16Manual512ProjectionPlan
)


def prepare_tle_sm90_projection(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> EagerT2048P256Manual512ProjectionPlan:
    """Prepare the admitted cluster16 H256 exact-shape implementation."""
    plan = _base.prepare_tle_sm90_projection(inputs, outputs)
    plan.metadata.update(
        {
            "T": TOKENS,
            "grid_ctas": GRID_CTAS,
            "num_m_tiles": NUM_M_TILES,
            "logical_grid_tiles": NUM_M_TILES,
            "cluster_ctas": CLUSTER_CTAS,
            "nonportable_cluster_size": True,
            "requires_cluster_capability_admission": True,
            "num_splits": NUM_SPLITS,
            "split_hidden": SPLIT_HIDDEN,
            "active_k64_blocks": SPLIT_HIDDEN // 64,
            "cta_mapping": "tile=pid//16, split=pid%16",
            "shape_specialization": "T=2048 H256 cluster16",
        }
    )
    return plan


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> EagerT2048P256Manual512ProjectionPlan:
    """Prepare the admitted full single-kernel operator."""
    return prepare_tle_sm90_projection(inputs, outputs)


__all__ = [
    "CLUSTER_CTAS",
    "EagerT2048P256Manual512ProjectionPlan",
    "GRID_CTAS",
    "MAX_SPLITS",
    "NUM_M_TILES",
    "NUM_SPLITS",
    "SPLIT_HIDDEN",
    "TOKENS",
    "prepare_tle_sm90_mega_mhc",
    "prepare_tle_sm90_projection",
]
