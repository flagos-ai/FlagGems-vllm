# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# GCU adaptation of ``mv`` (out = mat @ vec).
#
# The generic kernel materializes 2D broadcast tiles (``arange[:, None]`` row
# offsets + ``arange[None, :]`` column offsets for addresses, masks and the
# accumulator). On the GCU any 2D-broadcast value fails fresh compiles with the
# ``memref_ext.slice_*`` backend regression (compiler_issues/vllm_outer.md;
# the generic ``flaggems_vllm.ops.mv`` was an archived instance of that crash),
# and on the 2026-11-16 TopsRider stack the failure surfaces as a hard device
# abort (SIGABRT) instead of a Python exception - reproduced at (384, 384) fp16
# on the very first launch, which is what killed QA TCC-2323's
# ``mv_and_outer_benchmark`` (the ``outer`` param is collateral: the mv param
# aborts the pytest process before outer runs).
#
# Like the delivered ``_enflame/ops/outer.py`` rewrite, this version streams
# rows with 1D vectors only: each program owns BLOCK_M rows, walks every row
# in BLOCK_K-wide chunks, accumulates in fp32 and stores one scalar per row.
# No 2D tiles are ever formed, so the regression path is unreachable.
#
# GCU400 launch heuristics (following outer.py probe results, gcu400 24 SIP):
#   - CTA dispatch costs ~42-55ns each (serialized front-end), so the program
#     count is capped around 32 CTAs for small tasks / 128 for large ones;
#     BLOCK_M rows per program keeps grid.x far below the 65535 hardware limit.
#   - Live vector bytes per warp stay <= 16KB (the empirically safe envelope
#     from compiler_issues/vllm_outer.md): BLOCK_K = 4096 for 16-bit dtypes /
#     2048 for fp32, always num_warps=2 (about 4KB per vector per warp, with
#     roughly three live vectors: a, b, product; the accumulator is scalar).
#   - The vec chunk is re-read per row; it is tiny compared to the matrix and
#     is absorbed by cache, keeping gemv at its memory-bound floor.
#
# Numerics (probe-isolated, jobs j20261108105732 / this job's isolation probe):
# the GCU backend contracts ``acc += a * b`` on a vector accumulator into a
# 16-bit FMA when both operands originate from bf16/fp16 loads, silently
# rounding the products back to input precision (rel ~3.8e-3 bf16 / 2.2e-4
# fp16 on a 1024-term dot; fp32 loads are exact, casts and standalone
# products are exact, ``tl.sum(a * b)`` is exact). The kernel therefore never
# uses a loop-carried vector accumulator: each chunk is reduced with
# ``tl.sum(a * b)`` (fp32 tree reduction, exact) and the per-chunk scalars
# are accumulated in fp32, which matches native topsaten gemv bit-for-bit
# at probed shapes.


import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)

# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255.
_GRID_X_MAX = 65535
_BLOCK_K_MAX = 8192
_BLOCK_M_CAP = 64
_TARGET_CTAS = 128
_TARGET_CTAS_SMALL = 32
_SMALL_TASK_ELEMS = 1 << 23
_NW = 2


def _mv_blocks(n, m, itemsize):
    total = n * m
    if total >= _SMALL_TASK_ELEMS:
        target = _TARGET_CTAS
    else:
        target = _TARGET_CTAS_SMALL
    block_k = min(_BLOCK_K_MAX, triton.next_power_of_2(max(m, 1)))
    if itemsize >= 4 and block_k > 2048:
        # Keep live vector bytes per warp within the probed-safe envelope.
        block_k = 2048
    block_m = min(triton.cdiv(n, target), _BLOCK_M_CAP)
    block_m = max(block_m, triton.cdiv(n, _GRID_X_MAX), 1)
    return block_m, block_k


@libentry()
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    for i in tl.static_range(BLOCK_M):
        row = pid * BLOCK_M + i
        if row < N:
            total = 0.0
            row_base = row.to(tl.int64) * stride_an
            for k0 in range(0, M, BLOCK_K):
                cols = k0 + tl.arange(0, BLOCK_K)
                mask = cols < M
                a = tl.load(A + row_base + cols * stride_am, mask=mask, other=0.0).to(
                    tl.float32
                )
                b = tl.load(B + cols * stride_bm, mask=mask, other=0.0).to(tl.float32)
                # fp32 tree reduction per chunk; no loop-carried vector acc
                # (16-bit-sourced ``acc += a * b`` gets FMA-rounded, see
                # file header).
                total += tl.sum(a * b, axis=0)
            out_base = row.to(tl.int64) * stride_cn
            tl.store(C + out_base, total.to(C.type.element_ty))


def mv(inp, vec):
    logger.debug("GEMS_ENFLAME MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    if out.numel() == 0:
        return out
    BLOCK_M, BLOCK_K = _mv_blocks(N, M, inp.element_size())
    grid = (triton.cdiv(N, BLOCK_M),)
    with torch_device_fn.device(inp.device):
        mv_kernel[grid](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            num_warps=_NW,
        )
    return out
