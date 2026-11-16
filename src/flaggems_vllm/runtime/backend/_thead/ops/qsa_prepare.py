"""Fused QSA prepare kernel for Qwen4Exp."""

from flaggems_vllm.ops.qwen4.qsa_prepare import GENERIC_SPEC, LaunchSpec, build

# The scalar row path is kept for every size on this backend.
_THEAD_SPEC = LaunchSpec(
    name="thead",
    q_tiles=GENERIC_SPEC.q_tiles,
    main_tile_t=lambda _n: 1,
)

qsa_prepare = build(_THEAD_SPEC)

__all__ = ["qsa_prepare"]
