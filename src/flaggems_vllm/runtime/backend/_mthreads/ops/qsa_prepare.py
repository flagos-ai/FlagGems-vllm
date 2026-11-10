"""Fused QSA prepare kernel for Qwen4Exp."""

from flaggems_vllm.ops.qsa_prepare import GENERIC_SPEC, LaunchSpec, build

# Tiling only pays off once the grid is large enough to be launch-bound; below
# that the per-tile mask and the extra registers cost more than they save.
_MTHREADS_SPEC = LaunchSpec(
    name="mthreads",
    q_tiles=lambda n: (4, 4) if n > 4096 else GENERIC_SPEC.q_tiles(n),
    main_tile_t=lambda n: 8 if n >= 4096 else 1,
    num_warps=lambda n: 4 if n >= 4096 else 1,
)

qsa_prepare = build(_MTHREADS_SPEC)

__all__ = ["qsa_prepare"]
