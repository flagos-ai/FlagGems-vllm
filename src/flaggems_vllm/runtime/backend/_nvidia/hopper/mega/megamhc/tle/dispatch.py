"""Shape-strict dispatcher for the Hopper MegaMHC TLE candidates.

The dispatcher deliberately selects exactly one TLE kernel family and never
falls back to DeepGEMM CUDA or vLLM.  A missing shape is therefore observable
by the 60-case acceptance harness instead of being reported as a neutral
candidate result.

Kernel modules specialize several launch constants at import time.  Formal
benchmarks must consequently run one token bucket per fresh process.  This
module detects an attempted in-process bucket change and fails explicitly.
"""

from __future__ import annotations

import importlib
import os
from typing import Mapping


GRAPH_TOKENS = (
    1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
    112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
    208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336,
    352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512,
)

EAGER_TOKENS = (
    1028, 2048, 3076, 3805, 4100, 12991, 13005, 13021, 16384,
)

SMALL_SIMT_TOKENS = (1, 2, 4, 8)
PADDED_M64_TOKENS = (16, 24, 32, 40, 48, 56)
T64_TOKENS = (64,)
PADDED_M128_TILE_SPLIT64_TOKENS = (72, 80, 88, 96, 104, 112, 120, 128)
MID_ROLE2_PTX_SINKHORN_TOKENS = tuple(range(136, 257, 8))
LARGE_ROLE4_FORMAL_TOKENS = (
    272, 288, 304, 320, 336, 352, 368, 384, 480,
)
LARGE_SPLIT16_RS_FORMAL_TOKENS = (432,)
LARGE_SPLIT16_RS_T400_FORMAL_TOKENS = (400,)
LARGE_SPLIT16_RS_T416_FORMAL_TOKENS = (416,)
LARGE_SPLIT16_RS_T448_FORMAL_TOKENS = (448,)
LARGE_SPLIT16_RS_T464_FORMAL_TOKENS = (464,)
LARGE_SPLIT16_RS_T496_FORMAL_TOKENS = (496,)
LARGE_SPLIT16_RS_T512_FORMAL_TOKENS = (512,)


class MissingTLECaseError(RuntimeError):
    """Raised when no genuine TLE candidate exists for a manifest case."""


_PADDED_ENV = {
    "DG_TLE_MHC_T32_WGMMA_BM64_DOT": "1",
    "DG_TLE_MHC_T32_WGMMA_NUM_SPLITS": "128",
    "DG_TLE_MHC_T32_WGMMA_NUM_WARPS": "16",
    "DG_TLE_MHC_T32_WGMMA_STAGED_MATH_REGS": "128",
    "DG_TLE_MHC_T32_WGMMA_EXPLICIT_WGMMA": "1",
    "DG_TLE_MHC_T32_WGMMA_STAGED_WGMMA": "1",
    "DG_TLE_MHC_T32_WGMMA_WARP_OVERLAP": "1",
    "DG_TLE_MHC_T32_WGMMA_DIRECT_ROUTING_TAIL": "1",
    "DG_TLE_MHC_T32_WGMMA_DIRECT_TAIL_ILP4": "0",
    "DG_TLE_MHC_T32_WGMMA_PERSISTENT_WS": "1",
    "DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4": "1",
    "DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4_REDUCTION": "serial",
    "DG_TLE_MHC_T32_WGMMA_ROUTE4_WARP0_FINALIZE": "0",
    "DG_TLE_MHC_T32_WGMMA_SPLIT128_PAIR_ROUTE": "0",
    "DG_TLE_MHC_T32_WGMMA_EARLY_PRE_SYNC": "1",
    "DG_TLE_MHC_T32_WGMMA_PHASE_PROFILE": "0",
}

_T64_ENV = {
    "DG_TLE_MHC_T64_BM64_DOT": "1",
    "DG_TLE_MHC_T64_NUM_SPLITS": "128",
    "DG_TLE_MHC_T64_NUM_WARPS": "16",
    "DG_TLE_MHC_T64_STAGED_MATH_REGS": "128",
    "DG_TLE_MHC_T64_EXPLICIT_WGMMA": "1",
    "DG_TLE_MHC_T64_STAGED_WGMMA": "1",
    "DG_TLE_MHC_T64_WARP_OVERLAP": "1",
    "DG_TLE_MHC_T64_DIRECT_ROUTING_TAIL": "1",
    "DG_TLE_MHC_T64_DIRECT_TAIL_ILP4": "0",
    "DG_TLE_MHC_T64_PERSISTENT_WS": "1",
    "DG_TLE_MHC_T64_PERSISTENT_ROUTE4": "1",
    "DG_TLE_MHC_T64_PERSISTENT_ROUTE4_REDUCTION": "serial",
    "DG_TLE_MHC_T64_ROUTE4_WARP0_FINALIZE": "0",
    "DG_TLE_MHC_T64_SPLIT128_PAIR_ROUTE": "0",
    "DG_TLE_MHC_T64_EARLY_PRE_SYNC": "1",
    "DG_TLE_MHC_PHASE_PROFILE": "0",
}

_loaded_case: tuple[str, int] | None = None


def _set_exact_environment(values: Mapping[str, str]) -> None:
    for name, value in values.items():
        current = os.environ.get(name)
        if current not in (None, value):
            raise RuntimeError(
                f"{name}={current!r}, but the admitted TLE path requires "
                f"{value!r}"
            )
        os.environ[name] = value


def _tokens_from_inputs(inputs: Mapping[str, object]) -> int:
    x = inputs.get("x")
    shape = getattr(x, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("inputs['x'] must be a rank-2 tensor")
    return int(shape[0])


def candidate_family(tokens: int, execution_mode: str) -> str | None:
    """Return the genuine TLE family or ``None`` for a missing case."""
    if execution_mode not in ("cuda_graph", "eager"):
        raise ValueError("execution_mode must be 'cuda_graph' or 'eager'")
    if execution_mode == "eager":
        if tokens == 1028:
            return "eager-t1028-p256-manual512-hybrid"
        if tokens == 2048:
            return "eager-t2048-p256-manual512"
        if tokens == 3076:
            return "eager-t3076-p256-cluster16-n24"
        if tokens == 3805:
            return "eager-t3805-p1024-cluster4-n24"
        if tokens == 4100:
            return "eager-t4100-p1024-resident4-n24"
        if tokens == 12991:
            return "eager-t12991-p384-manual512-tma"
        if tokens == 13005:
            return "eager-t13005-p384-manual512-tma"
        if tokens == 13021:
            return "eager-t13021-p384-manual512-tma"
        if tokens == 16384:
            return "eager-t16384-p384-manual512-tma"
        return None
    if tokens in SMALL_SIMT_TOKENS:
        return "small-simt"
    if tokens in PADDED_M64_TOKENS:
        return "padded-m64-wgmma"
    if tokens in T64_TOKENS:
        return "t64-wgmma"
    if tokens in PADDED_M128_TILE_SPLIT64_TOKENS:
        return "padded-m128-tile-split64-wgmma"
    if tokens in MID_ROLE2_PTX_SINKHORN_TOKENS:
        return "mid-role2-ptx-sinkhorn-wgmma"
    if tokens in LARGE_SPLIT16_RS_T400_FORMAL_TOKENS:
        return "large-split16-rs-t400-raw-n24-wgmma"
    if tokens in LARGE_SPLIT16_RS_T416_FORMAL_TOKENS:
        return "large-split16-rs-t416-wgmma"
    if tokens in LARGE_SPLIT16_RS_T448_FORMAL_TOKENS:
        return "large-split16-rs-t448-wgmma"
    if tokens in LARGE_SPLIT16_RS_T464_FORMAL_TOKENS:
        return "large-split16-rs-t464-wgmma"
    if tokens in LARGE_SPLIT16_RS_T496_FORMAL_TOKENS:
        return "large-split16-rs-t496-wgmma"
    if tokens in LARGE_SPLIT16_RS_T512_FORMAL_TOKENS:
        return "large-split16-rs-t512-wgmma"
    if tokens in LARGE_SPLIT16_RS_FORMAL_TOKENS:
        return "large-split16-rs-wgmma"
    if tokens in LARGE_ROLE4_FORMAL_TOKENS:
        return "large-role4-k64-wgmma"
    return None


def _load_prepare(tokens: int):
    global _loaded_case

    execution_mode = "eager" if tokens in EAGER_TOKENS else "cuda_graph"
    family = candidate_family(tokens, execution_mode)
    if family is None:
        raise MissingTLECaseError(
            f"no TLE MegaMHC candidate for {execution_mode} T={tokens}; "
            "the case must be reported as missing"
        )
    case = (family, tokens)
    if _loaded_case not in (None, case):
        raise RuntimeError(
            "TLE kernels use import-time shape specialization; run each "
            f"manifest case in a fresh process (loaded={_loaded_case}, "
            f"requested={case})"
        )

    if family == "eager-t1028-p256-manual512-hybrid":
        _set_exact_environment(
            {
                "DG_TLE_MHC_EAGER_T1028_P256_MANUAL512": "1",
                "DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_TAIL": "1",
                "DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_TAIL_MODE": "x8",
                "DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_TAIL_PHASE": "3",
                "DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_MAXNREG": "64",
                "DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_FN_TMA": "0",
                "DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_ADMITTED": "1",
            }
        )
        module_name = (
            ".kernels.sm90_eager_t1028_p256_manual512_cpasync"
        )
    elif family == "eager-t2048-p256-manual512":
        _set_exact_environment(
            {
                "DG_TLE_MHC_EAGER_T2048_P256_MANUAL512": "1",
                "DG_TLE_MHC_EAGER_T2048_P256_MANUAL512_TAIL": "1",
                "DG_TLE_MHC_EAGER_T2048_P256_MANUAL512_TAIL_MODE": "x8",
                "DG_TLE_MHC_EAGER_T2048_P256_MANUAL512_TAIL_PHASE": "3",
                "DG_TLE_MHC_EAGER_T2048_P256_MANUAL512_MAXNREG": "64",
                "DG_TLE_MHC_EAGER_T2048_P256_MANUAL512_ADMITTED": "1",
            }
        )
        module_name = (
            ".kernels.sm90_eager_t2048_p256_manual512_cpasync"
        )
    elif family == "eager-t3076-p256-cluster16-n24":
        _set_exact_environment(
            {
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512": "1",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL": "1",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL_MODE": "x8",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL_PHASE": "3",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_HIDDEN_PHASE": "3",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL_CHUNK": "128",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_EXACT1024_TAIL": "1",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_FUSED4096_TAIL": "0",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_ROW_BARRIER": "0",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_GLOBAL_TILE_BARRIER": "0",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_MAXNREG": "64",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_FN_TMA": "0",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_ADMITTED": "1",
            }
        )
        module_name = (
            ".kernels.sm90_eager_t3076_p256_cluster16_manual512_cpasync"
        )
    elif family == "eager-t3805-p1024-cluster4-n24":
        _set_exact_environment(
            {
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512": "1",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL": "1",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL_MODE": "x8",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL_PHASE": "3",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_HIDDEN_PHASE": "3",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL_CHUNK": "128",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_EXACT1024_TAIL": "1",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_FUSED4096_TAIL": "0",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_ROW_BARRIER": "0",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_GLOBAL_TILE_BARRIER": "0",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_MAXNREG": "64",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_FN_TMA": "0",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_ADMITTED": "1",
            }
        )
        module_name = (
            ".kernels.sm90_eager_t3805_p1024_cluster4_manual512_cpasync"
        )
    elif family == "eager-t4100-p1024-resident4-n24":
        _set_exact_environment(
            {
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512": "1",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL": "1",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL_MODE": "x8",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL_PHASE": "3",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_HIDDEN_PHASE": "3",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_TAIL_CHUNK": "128",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_EXACT1024_TAIL": "1",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_FUSED4096_TAIL": "0",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_ROW_BARRIER": "0",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_GLOBAL_TILE_BARRIER": "1",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_MAXNREG": "64",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_FN_TMA": "0",
                "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_ADMITTED": "1",
            }
        )
        module_name = (
            ".kernels.sm90_eager_t4100_p1024_cluster4_manual512_cpasync"
        )
    elif family == "eager-t12991-p384-manual512-tma":
        _set_exact_environment(
            {
                "DG_TLE_MHC_EAGER_T12991_P384": "1",
                "DG_TLE_MHC_EAGER_T12991_CTA_MODE": "manual512",
                "DG_TLE_MHC_EAGER_T12991_ROOT_WARPS": "12",
                "DG_TLE_MHC_EAGER_T12991_PRODUCER": "ptx-quad",
                "DG_TLE_MHC_EAGER_T12991_ACCUM_MODE": "ptx-persistent",
                "DG_TLE_MHC_EAGER_T12991_FN_STAGE": "tma",
                "DG_TLE_MHC_EAGER_T12991_HANDOFF": "k64",
                "DG_TLE_MHC_EAGER_T12991_PROBE_PHASE": "1",
                "DG_TLE_MHC_EAGER_T12991_TAIL_WORK_PHASE": "3",
                "DG_TLE_MHC_EAGER_T12991_TAIL_BATCHES": "22",
                "DG_TLE_MHC_EAGER_T12991_TAIL_SCHEDULE": "route12",
                "DG_TLE_MHC_EAGER_T12991_MANUAL_HIDDEN": "shared-round",
                "DG_TLE_MHC_EAGER_T12991_MAXNREG": "64",
                "DG_TLE_MHC_EAGER_T12991_ADMITTED": "1",
            }
        )
        module_name = ".kernels.sm90_eager_t12991_p384_rs_wgmma"
    elif family == "eager-t13005-p384-manual512-tma":
        _set_exact_environment(
            {
                "DG_TLE_MHC_EAGER_T13005_P384": "1",
                "DG_TLE_MHC_EAGER_T13005_CTA_MODE": "manual512",
                "DG_TLE_MHC_EAGER_T13005_ROOT_WARPS": "12",
                "DG_TLE_MHC_EAGER_T13005_PRODUCER": "ptx-quad",
                "DG_TLE_MHC_EAGER_T13005_ACCUM_MODE": "ptx-persistent",
                "DG_TLE_MHC_EAGER_T13005_FN_STAGE": "tma",
                "DG_TLE_MHC_EAGER_T13005_HANDOFF": "k64",
                "DG_TLE_MHC_EAGER_T13005_PROBE_PHASE": "1",
                "DG_TLE_MHC_EAGER_T13005_TAIL_WORK_PHASE": "3",
                "DG_TLE_MHC_EAGER_T13005_TAIL_BATCHES": "22",
                "DG_TLE_MHC_EAGER_T13005_TAIL_SCHEDULE": "route12",
                "DG_TLE_MHC_EAGER_T13005_MANUAL_HIDDEN": "shared-round",
                "DG_TLE_MHC_EAGER_T13005_MAXNREG": "64",
                "DG_TLE_MHC_EAGER_T13005_ADMITTED": "1",
            }
        )
        module_name = ".kernels.sm90_eager_t13005_p384_rs_wgmma"
    elif family == "eager-t13021-p384-manual512-tma":
        _set_exact_environment(
            {
                "DG_TLE_MHC_EAGER_T13021_P384": "1",
                "DG_TLE_MHC_EAGER_T13021_CTA_MODE": "manual512",
                "DG_TLE_MHC_EAGER_T13021_ROOT_WARPS": "12",
                "DG_TLE_MHC_EAGER_T13021_PRODUCER": "ptx-quad",
                "DG_TLE_MHC_EAGER_T13021_ACCUM_MODE": "ptx-persistent",
                "DG_TLE_MHC_EAGER_T13021_FN_STAGE": "tma",
                "DG_TLE_MHC_EAGER_T13021_HANDOFF": "k64",
                "DG_TLE_MHC_EAGER_T13021_PROBE_PHASE": "1",
                "DG_TLE_MHC_EAGER_T13021_TAIL_WORK_PHASE": "3",
                "DG_TLE_MHC_EAGER_T13021_TAIL_BATCHES": "22",
                "DG_TLE_MHC_EAGER_T13021_TAIL_SCHEDULE": "route12",
                "DG_TLE_MHC_EAGER_T13021_MANUAL_HIDDEN": "shared-round",
                "DG_TLE_MHC_EAGER_T13021_MAXNREG": "64",
                "DG_TLE_MHC_EAGER_T13021_ADMITTED": "1",
            }
        )
        module_name = ".kernels.sm90_eager_t13021_p384_rs_wgmma"
    elif family == "eager-t16384-p384-manual512-tma":
        _set_exact_environment(
            {
                "DG_TLE_MHC_EAGER_T16384_P384": "1",
                "DG_TLE_MHC_EAGER_T16384_CTA_MODE": "manual512",
                "DG_TLE_MHC_EAGER_T16384_ROOT_WARPS": "12",
                "DG_TLE_MHC_EAGER_T16384_PRODUCER": "ptx-quad",
                "DG_TLE_MHC_EAGER_T16384_ACCUM_MODE": "ptx-persistent",
                "DG_TLE_MHC_EAGER_T16384_FN_STAGE": "tma",
                "DG_TLE_MHC_EAGER_T16384_HANDOFF": "k64",
                "DG_TLE_MHC_EAGER_T16384_PROBE_PHASE": "1",
                "DG_TLE_MHC_EAGER_T16384_TAIL_WORK_PHASE": "3",
                "DG_TLE_MHC_EAGER_T16384_TAIL_BATCHES": "22",
                "DG_TLE_MHC_EAGER_T16384_TAIL_SCHEDULE": "route12",
                "DG_TLE_MHC_EAGER_T16384_MANUAL_HIDDEN": "shared-round",
                "DG_TLE_MHC_EAGER_T16384_MAXNREG": "64",
                "DG_TLE_MHC_EAGER_T16384_ADMITTED": "1",
            }
        )
        module_name = ".kernels.sm90_eager_t16384_p384_rs_wgmma"
    elif family == "small-simt":
        module_name = ".kernels.sm90_small_family"
    elif family == "padded-m64-wgmma":
        _set_exact_environment(_PADDED_ENV)
        _set_exact_environment(
            {"DG_TLE_MHC_PADDED_WGMMA_LOGICAL_TOKENS": str(tokens)}
        )
        module_name = ".kernels.sm90_padded_m64_wgmma"
    elif family == "t64-wgmma":
        _set_exact_environment(_T64_ENV)
        module_name = ".fused_post_gemm_t64"
    elif family == "padded-m128-tile-split64-wgmma":
        _set_exact_environment(
            {
                "DG_TLE_MHC_PADDED_M128_TILE_SPLIT64": "1",
                "DG_TLE_MHC_PADDED_M128_TILE_SPLIT64_LOGICAL_TOKENS": str(
                    tokens
                ),
            }
        )
        module_name = ".kernels.sm90_padded_m128_tile_split64_wgmma"
    elif family == "mid-role2-ptx-sinkhorn-wgmma":
        _set_exact_environment(
            {
                "DG_TLE_MHC_MID_GRID_PIPE4_ROLE2": "1",
                "DG_TLE_MHC_MID_GRID_PIPE4_ROLE2_LOGICAL_TOKENS": str(
                    tokens
                ),
            }
        )
        module_name = ".kernels.sm90_mid_grid_pipe4_role2_ptx_sinkhorn_wgmma"
    elif family == "large-split16-rs-t400-raw-n24-wgmma":
        module_name = (
            ".kernels.sm90_large_graph_split16_rs_t400_raw_n24_wgmma"
        )
    elif family == "large-split16-rs-t416-wgmma":
        module_name = (
            ".kernels.sm90_large_graph_split16_rs_t416_nonstriped_wgmma"
        )
    elif family == "large-split16-rs-t448-wgmma":
        module_name = ".kernels.sm90_large_graph_split16_rs_t448_wgmma"
    elif family == "large-split16-rs-t464-wgmma":
        module_name = (
            ".kernels.sm90_large_graph_split16_rs_t464_nonstriped_wgmma"
        )
    elif family == "large-split16-rs-t496-wgmma":
        module_name = (
            ".kernels.sm90_large_graph_split16_rs_t496_nonstriped_wgmma"
        )
    elif family == "large-split16-rs-t512-wgmma":
        module_name = ".kernels.sm90_large_graph_split16_rs_t512_wgmma"
    elif family == "large-split16-rs-wgmma":
        _set_exact_environment(
            {
                "DG_TLE_MHC_LARGE_GRAPH_ROLE4": "1",
                "DG_TLE_MHC_LARGE_GRAPH_ROLE4_LOGICAL_TOKENS": str(
                    tokens
                ),
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
        )
        module_name = ".kernels.sm90_large_graph_split16_rs_wgmma"
    elif family == "large-role4-k64-wgmma":
        # T480 is the same zero-spill K64/pipe8 kernel structure, but its
        # strict formal winner uses a lower math-partition register budget
        # and non-striped token ownership.  Keep the selection shape-exact;
        # applying these knobs to the formally admitted T272--384 buckets
        # would invalidate their evidence.
        is_t480 = tokens == 480
        _set_exact_environment(
            {
                "DG_TLE_MHC_LARGE_GRAPH_ROLE4": "1",
                "DG_TLE_MHC_LARGE_GRAPH_ROLE4_LOGICAL_TOKENS": str(
                    tokens
                ),
                "DG_TLE_MHC_LARGE_GRAPH_TRUE_K64_STAGE": "1",
                "DG_TLE_MHC_LARGE_GRAPH_COEFF_CACHE": (
                    "0" if is_t480 else "1"
                ),
                "DG_TLE_MHC_LARGE_GRAPH_WEIGHT_CACHE": "0",
                "DG_TLE_MHC_LARGE_GRAPH_PTX_SINKHORN": "1",
                "DG_TLE_MHC_LARGE_GRAPH_NON_STRIPED_TAIL": (
                    "1" if is_t480 else "0"
                ),
                "DG_TLE_MHC_LARGE_GRAPH_MATH_REGS": (
                    "96" if is_t480 else "128"
                ),
                "DG_TLE_MHC_LARGE_GRAPH_EARLY_PRE_SYNC": "1",
                "DG_TLE_MHC_LARGE_GRAPH_PROBE_PHASE": "4",
                "DG_TLE_MHC_LARGE_GRAPH_ROLE_OVERLAP": "0",
                "DG_TLE_MHC_LARGE_GRAPH_MANUAL_ROLE4_TAIL": "0",
                "DG_TLE_MHC_LARGE_GRAPH_PTX_BF16X8_TAIL": "0",
                "DG_TLE_MHC_LARGE_GRAPH_FAST_GRID_BARRIER": "0",
                "DG_TLE_MHC_LARGE_GRAPH_EXACT_T448": "0",
                "DG_TLE_MHC_LARGE_GRAPH_EXACT_T512": "0",
            }
        )
        module_name = ".kernels.sm90_large_graph_role4_k32_wgmma"
    else:
        raise RuntimeError(f"unhandled admitted TLE family {family!r}")

    module = importlib.import_module(module_name, package=__package__)
    _loaded_case = case
    return module.prepare_tle_sm90_mega_mhc


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
):
    """Prepare exactly one admitted Graph candidate without fallback."""
    tokens = _tokens_from_inputs(inputs)
    return _load_prepare(tokens)(inputs, outputs)


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
