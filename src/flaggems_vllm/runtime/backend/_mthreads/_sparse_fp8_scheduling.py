# Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Optional, Tuple

import torch
import triton
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources


def group_short_pages(
    rows: int, pages: int, programs: int
) -> Optional[Tuple[int, int, int, int]]:
    groups = math.gcd(rows, programs)
    group_rows, group_programs = rows // groups, programs // groups
    if group_rows == 1 or pages < group_programs:
        return None
    max_pages = triton.cdiv(group_rows * pages, group_programs)
    row_programs = triton.cdiv(pages, max_pages)
    row_slack = row_programs * max_pages - pages
    if row_slack == 0:
        return None
    # Use spare page capacity to align queries without lengthening the busiest CTA.
    spare_pages = group_programs * max_pages - group_rows * pages
    prefix_rows = min(group_rows - 1, spare_pages // row_slack)
    return (
        (group_rows, group_programs, prefix_rows, row_programs) if prefix_rows else None
    )


def prepare_persistent_combine(
    q,
    kv,
    qs,
    ks,
    indices,
    partial,
    stats,
    partial_flags,
    output,
    lse,
    scale,
    splits,
    programs,
    groups,
    repair,
    head_groups,
):
    from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_pipeline import (
        combine_persistent_sparse_fp8,
    )

    rows, topk = q.shape[0], indices.shape[-1]
    tokens = kv.shape[0] * 64
    args = (
        partial,
        stats,
        partial_flags,
        output,
        lse,
        rows,
        topk,
        splits,
        programs,
        triton.next_power_of_2(splits * head_groups),
        groups,
        head_groups,
    )
    combine_heads, combine_warps = 16, 8
    if rows == 16 and topk >= 4096:
        combine_heads, combine_warps = 32, 16
    elif rows <= 4 and topk >= 4096:
        combine_heads, combine_warps = 8, 4
    options = dict(
        num_warps=combine_warps,
        num_stages=1,
        enable_backend_opt=True,
        enable_llc_opt=True,
    )
    grid = (rows, 64 // combine_heads, 1)
    combine_args = args + (
        combine_heads,
        None,
        q,
        kv,
        qs,
        ks,
        indices,
        tokens,
        scale,
        repair,
    )
    try:
        kernel = combine_persistent_sparse_fp8.warmup(
            *combine_args, grid=grid, **options
        )
        return kernel[grid], combine_args, None
    except (CompilationError, OutOfResources, RuntimeError) as error:
        logging.getLogger(__name__).debug(
            "Sparse FP8 fused combine unavailable: %s", error
        )
    flags = torch.empty((rows, 1), dtype=torch.int32, device=q.device)
    grid = (rows, 8, 1)
    options["num_warps"] = 8
    combine_args = args + (8, flags, None, None, None, None, None, 0, 0.0, None)
    kernel = combine_persistent_sparse_fp8.warmup(*combine_args, grid=grid, **options)
    return kernel[grid], combine_args, flags
