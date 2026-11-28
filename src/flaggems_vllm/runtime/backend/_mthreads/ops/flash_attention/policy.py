# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Metadata-only candidate construction, shared by tuning and validation."""

from dataclasses import dataclass, replace

from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.scheduling import (
    AttentionPlan,
)

ROUTES = (
    "generic",
    "decode_tile",
    "bounded",
    "tle",
    "descriptor",
    "single_kv",
    "tle_decode",
)
FAMILIES = ("direct", "packed_decode", "compact", "split_kv", "split_decode")
MAX_CANDIDATES = 32
MAX_PARTIAL_BYTES = 64 * 1024 * 1024
COMPILER_ENVIRONMENT = (
    "TRITON_MUSA_ENABLE_LLC_OPT",
    "TRITON_MUSA_ENABLE_LLVM_SLP",
    "TRITON_MUSA_VERSION",
    "TRITON_DEFAULT_FP_FUSION",
)


def compiler_environment(environ):
    return tuple((name, environ.get(name, "unset")) for name in COMPILER_ENVIRONMENT)


@dataclass(frozen=True)
class Candidate:
    route: str
    plan: AttentionPlan
    gather_tile: int = 0
    gather_warps: int = 0
    pack: int = 0
    compact: int = -1

    def values(self):
        return dict(
            ROUTE=ROUTES.index(self.route),
            FAMILY=FAMILIES.index(self.plan.family),
            BM=self.plan.block_m,
            BN=self.plan.block_n,
            WARPS=self.plan.num_warps,
            STAGES=self.plan.num_stages,
            SPLITS=self.plan.num_splits,
            HEAD_PACK=self.plan.head_pack,
            GATHER_TILE=self.gather_tile,
            GATHER_WARPS=self.gather_warps,
            PACK=self.pack,
            COMPACT=self.compact,
        )


@dataclass(frozen=True)
class Problem:
    maxq: int
    maxk: int
    total: int
    heads: int
    groups: int
    dim: int
    batch: int
    routes: tuple


def candidates(problem, default, templates):
    """Keep the heuristic seed and each legal family before parameter neighbours."""
    result = [default]
    for template in templates:
        values = template.kwargs
        route = (
            default.route if values.get("ROUTE", -1) < 0 else ROUTES[values["ROUTE"]]
        )
        if route not in problem.routes:
            continue
        plan = default.plan
        if route == "generic":
            splits = values.get("SPLITS", plan.num_splits)
            if splits > 1:
                family = (
                    "split_decode"
                    if problem.maxq == 1 and problem.groups <= 16
                    else "split_kv"
                )
            elif problem.maxq == 1 and problem.groups <= 16:
                family = "packed_decode"
            else:
                family = (
                    "compact"
                    if values.get("COMPACT", int(plan.family == "compact")) == 1
                    else "direct"
                )
            plan = replace(
                plan,
                family=family,
                block_m=values.get("BM", plan.block_m),
                block_n=values.get("BN", plan.block_n),
                num_warps=values.get("WARPS", plan.num_warps),
                num_stages=values.get("STAGES", plan.num_stages),
                num_splits=splits,
                head_pack=values.get("HEAD_PACK", plan.head_pack),
            )
            if not legal_plan(problem, plan):
                continue
        elif route == "single_kv":
            plan = replace(
                plan,
                block_m=values.get("BM", plan.block_m),
                num_warps=values.get("WARPS", plan.num_warps),
            )
            if plan.block_m not in (0, 1, 4, 8, 16) or plan.num_warps not in (4, 8):
                continue
            if plan.block_m == 0 and plan.num_warps != 4:
                continue
        elif route == "tle_decode":
            plan = replace(
                plan,
                block_m=values.get("BM", 32),
                num_warps=values.get("WARPS", 4),
            )
            if plan.block_m != 32 or plan.num_warps != 4:
                continue
        candidate = Candidate(
            route,
            plan,
            values.get("GATHER_TILE", 0),
            values.get("GATHER_WARPS", 0),
            values.get("PACK", 0),
            values.get("COMPACT", -1),
        )
        if candidate.pack and (problem.groups % candidate.pack or candidate.pack > 8):
            continue
        if candidate.gather_tile not in (
            0,
            16,
            32,
            64,
        ) or candidate.gather_warps not in (0, 4, 8):
            continue
        if route not in ("bounded", "tle", "descriptor") and (
            candidate.gather_tile or candidate.gather_warps or candidate.pack
        ):
            continue
        if route != "bounded" and candidate.pack:
            continue
        if route != "bounded" and route != "generic" and candidate.compact != -1:
            continue
        if candidate not in result:
            result.append(candidate)
        if len(result) == MAX_CANDIDATES:
            break
    return result


def legal_plan(problem, plan):
    if plan.block_m not in (16, 32, 64) or plan.block_n not in (32, 64):
        return False
    if plan.num_warps not in (4, 8) or plan.num_stages not in (1, 2):
        return False
    if plan.head_pack not in (1, 4, 8) or problem.groups % plan.head_pack:
        return False
    if plan.num_splits not in (1, 2, 4, 8, 16, 32):
        return False
    if plan.num_splits > 1:
        if problem.maxk < 8192 or problem.total > 256:
            return False
        if (
            plan.num_splits * problem.total * problem.heads * (problem.dim + 1) * 4
            > MAX_PARTIAL_BYTES
        ):
            return False
    if plan.family == "compact" and not 1 < problem.batch <= 128:
        return False
    return True


def tensor_identity(tensor):
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        tensor.storage_offset(),
    )
