"""Shape-strict dispatcher for the Hopper MegaMHC TLE candidates.

The dispatcher selects exactly one TLE kernel family and never falls back to
DeepGEMM CUDA or vLLM. Kernel modules specialize several launch constants at
import time, so an in-process bucket change fails explicitly.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

GRAPH_TOKENS = (1, 2, 4, 8, 16, *range(24, 257, 8), *range(272, 513, 16))
EAGER_TOKENS = (1028, 2048, 3076, 3805, 4100, 12991, 13005, 13021, 16384)

SMALL_SIMT_TOKENS = (1, 2, 4, 8)
PADDED_M64_TOKENS = (16, 24, 32, 40, 48, 56)
T64_TOKENS = (64,)
PADDED_M128_TILE_SPLIT64_TOKENS = (72, 80, 88, 96, 104, 112, 120, 128)
MID_ROLE2_PTX_SINKHORN_TOKENS = tuple(range(136, 257, 8))
LARGE_ROLE4_FORMAL_TOKENS = (272, 288, 304, 320, 336, 352, 368, 384, 480)
LARGE_SPLIT16_RS_FORMAL_TOKENS = (432,)
LARGE_SPLIT16_RS_T400_FORMAL_TOKENS = (400,)
LARGE_SPLIT16_RS_T416_FORMAL_TOKENS = (416,)
LARGE_SPLIT16_RS_T448_FORMAL_TOKENS = (448,)
LARGE_SPLIT16_RS_T464_FORMAL_TOKENS = (464,)
LARGE_SPLIT16_RS_T496_FORMAL_TOKENS = (496,)
LARGE_SPLIT16_RS_T512_FORMAL_TOKENS = (512,)


class MissingTLECaseError(RuntimeError):
    """Raised when no genuine TLE candidate exists for a manifest case."""


def _prefixed(prefix: str, suffix_values: Mapping[str, str]) -> dict[str, str]:
    return {
        prefix + (f"_{suffix}" if suffix else ""): value
        for suffix, value in suffix_values.items()
    }


# fmt: off
_PERSISTENT_SUFFIXES = {
    "BM64_DOT": "1", "NUM_SPLITS": "128", "NUM_WARPS": "16",
    "STAGED_MATH_REGS": "128", "EXPLICIT_WGMMA": "1", "STAGED_WGMMA": "1",
    "WARP_OVERLAP": "1", "DIRECT_ROUTING_TAIL": "1", "DIRECT_TAIL_ILP4": "0",
    "PERSISTENT_WS": "1", "PERSISTENT_ROUTE4": "1",
    "PERSISTENT_ROUTE4_REDUCTION": "serial", "ROUTE4_WARP0_FINALIZE": "0",
    "SPLIT128_PAIR_ROUTE": "0", "EARLY_PRE_SYNC": "1",
}
_P256_MANUAL_SUFFIXES = {
    "": "1", "TAIL": "1", "TAIL_MODE": "x8", "TAIL_PHASE": "3",
    "MAXNREG": "64", "ADMITTED": "1",
}
_P256_CLUSTER_SUFFIXES = {
    "": "1", "TAIL": "1", "TAIL_MODE": "x8", "TAIL_PHASE": "3",
    "HIDDEN_PHASE": "3", "TAIL_CHUNK": "128", "EXACT1024_TAIL": "1",
    "FUSED4096_TAIL": "0", "ROW_BARRIER": "0", "MAXNREG": "64",
    "FN_TMA": "0", "ADMITTED": "1",
}
_P384_SUFFIXES = {
    "P384": "1", "CTA_MODE": "manual512", "ROOT_WARPS": "12",
    "PRODUCER": "ptx-quad", "ACCUM_MODE": "ptx-persistent", "FN_STAGE": "tma",
    "HANDOFF": "k64", "PROBE_PHASE": "1", "TAIL_WORK_PHASE": "3",
    "TAIL_BATCHES": "22", "TAIL_SCHEDULE": "route12",
    "MANUAL_HIDDEN": "shared-round", "MAXNREG": "64", "ADMITTED": "1",
}
_LARGE_COMMON_SUFFIXES = {
    "ROLE4": "1", "TRUE_K64_STAGE": "1", "WEIGHT_CACHE": "0",
    "PTX_SINKHORN": "1", "EARLY_PRE_SYNC": "1", "PROBE_PHASE": "4",
    "ROLE_OVERLAP": "0", "MANUAL_ROLE4_TAIL": "0", "PTX_BF16X8_TAIL": "0",
    "FAST_GRID_BARRIER": "0", "EXACT_T448": "0", "EXACT_T512": "0",
}
# fmt: on


def _no_environment(tokens: int) -> Mapping[str, str]:
    return {}


def _persistent_environment(tokens: int) -> Mapping[str, str]:
    prefix = "DG_TLE_MHC_T32_WGMMA" if tokens != 64 else "DG_TLE_MHC_T64"
    values = _prefixed(prefix, _PERSISTENT_SUFFIXES)
    phase_name = (
        "DG_TLE_MHC_T32_WGMMA_PHASE_PROFILE"
        if tokens != 64
        else "DG_TLE_MHC_PHASE_PROFILE"
    )
    values[phase_name] = "0"
    if tokens != 64:
        values["DG_TLE_MHC_PADDED_WGMMA_LOGICAL_TOKENS"] = str(tokens)
    return values


def _p256_manual512_environment(tokens: int) -> Mapping[str, str]:
    prefix = f"DG_TLE_MHC_EAGER_T{tokens}_P256_MANUAL512"
    values = _prefixed(prefix, _P256_MANUAL_SUFFIXES)
    if tokens == 1028:
        values[f"{prefix}_FN_TMA"] = "0"
    return values


def _p256_cluster_environment(tokens: int) -> Mapping[str, str]:
    prefix = "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512"
    values = _prefixed(prefix, _P256_CLUSTER_SUFFIXES)
    values[f"{prefix}_GLOBAL_TILE_BARRIER"] = "1" if tokens == 4100 else "0"
    return values


def _p384_environment(tokens: int) -> Mapping[str, str]:
    return _prefixed(f"DG_TLE_MHC_EAGER_T{tokens}", _P384_SUFFIXES)


def _logical_environment(tokens: int, prefix: str) -> Mapping[str, str]:
    return {prefix: "1", f"{prefix}_LOGICAL_TOKENS": str(tokens)}


def _large_environment(tokens: int, *, split16: bool) -> Mapping[str, str]:
    prefix = "DG_TLE_MHC_LARGE_GRAPH"
    values = _prefixed(prefix, _LARGE_COMMON_SUFFIXES)
    values[f"{prefix}_ROLE4_LOGICAL_TOKENS"] = str(tokens)
    if split16:
        values.update(
            _prefixed(
                prefix,
                {
                    "COEFF_CACHE": "0",
                    "NON_STRIPED_TAIL": "0",
                    "MATH_REGS": "128",
                    "ALLOCATED_ROLE_BARRIERS": "0",
                    "SHARED_ROLE_FLAGS": "0",
                    "TILE_LOCAL_BARRIER": "0",
                    "SPLIT16_RS_MATH_REGS": "128",
                },
            )
        )
    else:
        is_t480 = tokens == 480
        values.update(
            _prefixed(
                prefix,
                {
                    "COEFF_CACHE": "0" if is_t480 else "1",
                    "NON_STRIPED_TAIL": "1" if is_t480 else "0",
                    "MATH_REGS": "96" if is_t480 else "128",
                },
            )
        )
    return values


def _large_split16_environment(tokens: int) -> Mapping[str, str]:
    return _large_environment(tokens, split16=True)


def _large_role4_environment(tokens: int) -> Mapping[str, str]:
    return _large_environment(tokens, split16=False)


def _padded_m128_environment(tokens: int) -> Mapping[str, str]:
    return _logical_environment(tokens, "DG_TLE_MHC_PADDED_M128_TILE_SPLIT64")


def _mid_environment(tokens: int) -> Mapping[str, str]:
    return _logical_environment(tokens, "DG_TLE_MHC_MID_GRID_PIPE4_ROLE2")


@dataclass(frozen=True)
class _CaseSpec:
    family: str
    module: str
    environment: Callable[[int], Mapping[str, str]] = _no_environment


def _family_specs(
    tokens: tuple[int, ...],
    family: str,
    module: str,
    environment: Callable[[int], Mapping[str, str]] = _no_environment,
) -> dict[int, _CaseSpec]:
    spec = _CaseSpec(family, module, environment)
    return {token: spec for token in tokens}


# Complete production matrix: token, family, module, and environment stay together.
# fmt: off
_EAGER_CASES = {
    1028: _CaseSpec("eager-t1028-p256-manual512-hybrid",
                    ".kernels.sm90_eager_t1028_p256_manual512_cpasync", _p256_manual512_environment),
    2048: _CaseSpec("eager-t2048-p256-manual512",
                    ".kernels.sm90_eager_t2048_p256_manual512_cpasync", _p256_manual512_environment),
    3076: _CaseSpec("eager-t3076-p256-cluster16-n24",
                    ".kernels.sm90_eager_t3076_p256_cluster16_manual512_cpasync", _p256_cluster_environment),
    3805: _CaseSpec("eager-t3805-p1024-cluster4-n24",
                    ".kernels.sm90_eager_t3805_p1024_cluster4_manual512_cpasync", _p256_cluster_environment),
    4100: _CaseSpec("eager-t4100-p1024-resident4-n24",
                    ".kernels.sm90_eager_t4100_p1024_cluster4_manual512_cpasync", _p256_cluster_environment),
    **{
        token: _CaseSpec(f"eager-t{token}-p384-manual512-tma",
                         f".kernels.sm90_eager_t{token}_p384_rs_wgmma", _p384_environment)
        for token in (12991, 13005, 13021, 16384)
    },
}
_GRAPH_CASES = {
    **_family_specs(SMALL_SIMT_TOKENS, "small-simt", ".kernels.sm90_small_family"),
    **_family_specs(PADDED_M64_TOKENS, "padded-m64-wgmma", ".kernels.sm90_padded_m64_wgmma", _persistent_environment),
    **_family_specs(T64_TOKENS, "t64-wgmma", ".fused_post_gemm_t64", _persistent_environment),
    **_family_specs(PADDED_M128_TILE_SPLIT64_TOKENS, "padded-m128-tile-split64-wgmma",
                    ".kernels.sm90_padded_m128_tile_split64_wgmma", _padded_m128_environment),
    **_family_specs(MID_ROLE2_PTX_SINKHORN_TOKENS, "mid-role2-ptx-sinkhorn-wgmma",
                    ".kernels.sm90_mid_grid_pipe4_role2_ptx_sinkhorn_wgmma", _mid_environment),
    **_family_specs(LARGE_ROLE4_FORMAL_TOKENS, "large-role4-k64-wgmma",
                    ".kernels.sm90_large_graph_role4_k32_wgmma", _large_role4_environment),
    400: _CaseSpec("large-split16-rs-t400-raw-n24-wgmma", ".kernels.sm90_large_graph_split16_rs_t400_raw_n24_wgmma"),
    416: _CaseSpec("large-split16-rs-t416-wgmma", ".kernels.sm90_large_graph_split16_rs_t416_nonstriped_wgmma"),
    432: _CaseSpec("large-split16-rs-wgmma", ".kernels.sm90_large_graph_split16_rs_wgmma", _large_split16_environment),
    448: _CaseSpec("large-split16-rs-t448-wgmma", ".kernels.sm90_large_graph_split16_rs_t448_wgmma"),
    464: _CaseSpec("large-split16-rs-t464-wgmma", ".kernels.sm90_large_graph_split16_rs_t464_nonstriped_wgmma"),
    496: _CaseSpec("large-split16-rs-t496-wgmma", ".kernels.sm90_large_graph_split16_rs_t496_nonstriped_wgmma"),
    512: _CaseSpec("large-split16-rs-t512-wgmma", ".kernels.sm90_large_graph_split16_rs_t512_wgmma"),
}
# fmt: on

_CASES_BY_MODE = {"cuda_graph": _GRAPH_CASES, "eager": _EAGER_CASES}
_loaded_case: tuple[str, int] | None = None


def _set_exact_environment(values: Mapping[str, str]) -> None:
    for name, value in values.items():
        current = os.environ.get(name)
        if current not in (None, value):
            raise RuntimeError(
                f"{name}={current!r}, but the admitted TLE path requires {value!r}"
            )
        os.environ[name] = value


def _tokens_from_inputs(inputs: Mapping[str, object]) -> int:
    x = inputs.get("x")
    shape = getattr(x, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("inputs['x'] must be a rank-2 tensor")
    return int(shape[0])


def candidate_family(tokens: int, execution_mode: str) -> str | None:
    """Return the genuine TLE family or None for a missing case."""
    if execution_mode not in _CASES_BY_MODE:
        raise ValueError("execution_mode must be 'cuda_graph' or 'eager'")
    spec = _CASES_BY_MODE[execution_mode].get(tokens)
    return None if spec is None else spec.family


def _load_prepare(tokens: int):
    global _loaded_case

    execution_mode = "eager" if tokens in EAGER_TOKENS else "cuda_graph"
    spec = _CASES_BY_MODE[execution_mode].get(tokens)
    if spec is None:
        raise MissingTLECaseError(
            f"no TLE MegaMHC candidate for {execution_mode} T={tokens}; "
            "the case must be reported as missing"
        )
    case = (spec.family, tokens)
    if _loaded_case not in (None, case):
        raise RuntimeError(
            "TLE kernels use import-time shape specialization; run each "
            f"manifest case in a fresh process (loaded={_loaded_case}, "
            f"requested={case})"
        )
    _set_exact_environment(spec.environment(tokens))
    module = importlib.import_module(spec.module, package=__package__)
    _loaded_case = case
    return module.prepare_tle_sm90_mega_mhc


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
):
    """Prepare exactly one admitted candidate without fallback."""
    return _load_prepare(_tokens_from_inputs(inputs))(inputs, outputs)


__all__ = [
    "EAGER_TOKENS",
    "GRAPH_TOKENS",
    "LARGE_SPLIT16_RS_FORMAL_TOKENS",
    "LARGE_SPLIT16_RS_T400_FORMAL_TOKENS",
    "LARGE_SPLIT16_RS_T416_FORMAL_TOKENS",
    "LARGE_SPLIT16_RS_T448_FORMAL_TOKENS",
    "LARGE_SPLIT16_RS_T464_FORMAL_TOKENS",
    "LARGE_SPLIT16_RS_T496_FORMAL_TOKENS",
    "LARGE_SPLIT16_RS_T512_FORMAL_TOKENS",
    "LARGE_ROLE4_FORMAL_TOKENS",
    "MissingTLECaseError",
    "candidate_family",
    "prepare_tle_sm90_mega_mhc",
]
