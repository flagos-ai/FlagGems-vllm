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

# GCU adaptation of ``outer``.
#
# The generic implementation broadcasts ``(M, 1) x (1, N)`` through the
# pointwise ``mul``/``mv`` ops. On the GCU that path has two problems:
#   1. the pointwise wrapper derives ``num_ctas = min(65536, num_tiles)``
#      from the NVIDIA-fallback codegen config, one CTA above the GCU
#      grid.x hardware limit (65535), so bench shapes such as
#      (1024, 65536) fail to launch;
#   2. fresh compiles of any 2D-broadcast pointwise/mv kernel currently
#      hit the ``memref_ext.slice_*`` backend regression (see
#      compiler_issues/vllm_outer.md), so the generic path only works
#      from stale triton cache entries.
# Both are avoided by dedicated row-streaming kernels that never
# materialize a 2D broadcast value (scalar x vector / vector x vector
# only) and launch within the GCU grid limits.
#
# Forward tiling: one CTA streams BLOCK_M rows of BLOCK_N-wide chunks. The
# y vector is loaded once per chunk and reused across all BLOCK_M rows
# (row-loop inside the chunk loop), instead of once per row: at (4096, 4096)
# that removes up to ~64MB of redundant y re-reads per launch.
# GCU400 launch heuristics (job j20261108105732 probes, gcu400 24 SIP):
#   - CTA dispatch costs ~42-55ns each (serialized front-end), so more CTAs
#     than needed is a pure loss: ~32 CTAs wins for small outputs (< 2**23
#     elements, e.g. (1024, 1024)), ~128 CTAs saturates the store path for
#     large ones ((4096, 4096), (1024, 65536)).
#   - BLOCK_N must cover as much of a row as possible: chunking a row into
#     smaller BLOCK_N stores is monotonically slower (BN=1024 at (4096, 4096)
#     is ~2.5x slower than BN=4096). BLOCK_N is capped at 8192 because 16384
#     hits a backend compiler regression (see compiler_issues/vllm_outer.md).
#   - num_warps: 2 for BLOCK_N <= 4096 (fastest and compile-safe for all
#     dtypes), 4 for BLOCK_N == 8192 (fp32 + num_warps < 4 + BLOCK_N 8192
#     triggers the memref_ext.slice_pad_start compile failure, same archive).
#
# Forward even-tile fast path (job j20261111121046 probes, gcu400):
# When M % BLOCK_M == 0 and N % BLOCK_N == 0 every mask and row guard is
# statically true, so the kernel drops them (masked-store predicate tax) and
# prefetches the x scalars one row ahead of each store (breaks the
# scalar-load -> store serialization chain). Measured kernel-only medians
# (do_bench w250/r100, vs the legacy masked kernel):
#   (1024,1024) fp16/bf16: 11.3 -> 7.0/6.9us (BM=32, nw2)
#   (1024,1024) fp32:      11.2 -> 7.9us    (BM=16, nw2)
#   (4096,4096) fp16/bf16: 32.2 -> 21.9us   (BM=16, nw2, i32 offsets)
#   (4096,4096) fp32:      34.9 -> 29.9us   (BM=16, nw4, i32 offsets; the
#                          y-broadcast store-only floor is 29.1us)
#   (1024,65536) fp16:     61.2 -> 56.3us   (BM=8, nw4, legacy BM rule)
# BM caps are per-N-band because the store path likes different CTA counts
# per row width / dtype (16-bit mid-N wants grid=256 at (4096,4096) but
# grid=32 at (1024,1024); fp32 swaps the two). i32 row offsets only pay off
# on the mid-N band (N in (1024, 4096]) and are guarded by M*N < 2**31;
# elsewhere they are neutral to 35% slower, so the fast path keeps i64.
# Shapes that do not tile evenly keep the legacy masked kernel verbatim.


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

_FWD_BLOCK_N_MAX = 8192
_FWD_BLOCK_M_CAP = 64
_FWD_TARGET_CTAS = 128
_FWD_TARGET_CTAS_SMALL = 32
_FWD_SMALL_TASK_ELEMS = 1 << 23
_FWD_NW_SPLIT = 4096
_FWD_I32_LIMIT = 1 << 31
_BWD_BLOCK_N = 512


def _fwd_blocks(m, n, itemsize):
    total = m * n
    if total >= _FWD_SMALL_TASK_ELEMS:
        target = _FWD_TARGET_CTAS
    else:
        target = _FWD_TARGET_CTAS_SMALL
    block_n = min(_FWD_BLOCK_N_MAX, triton.next_power_of_2(max(n, 1)))
    legacy_nw = 2 if block_n <= _FWD_NW_SPLIT else 4
    block_m = min(triton.cdiv(m, target), _FWD_BLOCK_M_CAP)
    block_m = max(block_m, triton.cdiv(m, _GRID_X_MAX), 1)
    if m % block_m != 0 or n % block_n != 0:
        # Non-even tiling: legacy masked kernel, legacy launch config.
        return block_m, block_n, legacy_nw, False, False

    # Even tiling: tuned fast-path config (probe-verified combos only).
    two_byte = itemsize <= 2
    if n > 4096:
        # Wide rows: legacy BM rule already optimal ((1024, 65536) class).
        return block_m, block_n, 4, True, False
    if n > 1024:
        # Mid rows: BM=16 wins big for 16-bit (32.2 -> 21.9us at (4096,4096))
        # and for fp32 with nw4 (34.9 -> 29.9us); i32 offsets pay off here.
        bm = max(min(triton.cdiv(m, target), 16), triton.cdiv(m, _GRID_X_MAX), 1)
        if m % bm != 0:
            return block_m, block_n, legacy_nw, True, False
        if total < _FWD_I32_LIMIT:
            num_warps = 2 if (two_byte or block_n <= 2048) else 4
            return bm, block_n, num_warps, True, True
        num_warps = 4 if two_byte else 2
        return bm, block_n, num_warps, True, False
    # Narrow rows: 16-bit wants BM=32 (grid 32 at (1024,1024)), fp32 BM=16.
    cap = 32 if two_byte else 16
    bm = max(min(triton.cdiv(m, target), cap), triton.cdiv(m, _GRID_X_MAX), 1)
    if m % bm != 0:
        return block_m, block_n, legacy_nw, True, False
    return bm, block_n, 2, True, False


@libentry()
@triton.jit
def outer_fwd_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN: tl.constexpr,
    I32: tl.constexpr,
):
    pid = tl.program_id(0)
    if EVEN:
        # Even tiling: no masks, no row guards; x scalars are prefetched one
        # row ahead of each store (probe-verified body, see file header).
        row0 = pid * BLOCK_M
        for j0 in range(0, N, BLOCK_N):
            cols = j0 + tl.arange(0, BLOCK_N)
            yv = tl.load(y_ptr + cols)
            nxt = tl.load(x_ptr + row0)
            for i in tl.static_range(BLOCK_M):
                cur = nxt
                if i + 1 < BLOCK_M:
                    nxt = tl.load(x_ptr + row0 + i + 1)
                if I32:
                    tl.store(out_ptr + (row0 + i) * N + cols, cur * yv)
                else:
                    out_base = (row0 + i).to(tl.int64) * N
                    tl.store(out_ptr + out_base + cols, cur * yv)
    else:
        for j0 in range(0, N, BLOCK_N):
            cols = j0 + tl.arange(0, BLOCK_N)
            mask = cols < N
            yv = tl.load(y_ptr + cols, mask=mask, other=0.0)
            for i in tl.static_range(BLOCK_M):
                row = pid * BLOCK_M + i
                if row < M:
                    xv = tl.load(x_ptr + row)
                    out_base = row.to(tl.int64) * N
                    tl.store(out_ptr + out_base + cols, xv * yv, mask=mask)


@libentry()
@triton.jit
def outer_bwd_inp_kernel(
    og_ptr,
    w_ptr,
    ig_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    for i in tl.static_range(BLOCK_M):
        row = pid * BLOCK_M + i
        if row < M:
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            og_base = row.to(tl.int64) * N
            for j0 in range(0, N, BLOCK_N):
                cols = j0 + tl.arange(0, BLOCK_N)
                mask = cols < N
                g = tl.load(og_ptr + og_base + cols, mask=mask, other=0.0).to(
                    tl.float32
                )
                w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                acc += g * w
            tl.store(ig_ptr + row, tl.sum(acc, axis=0).to(ig_ptr.type.element_ty))


@libentry()
@triton.jit
def outer_bwd_weight_kernel(
    og_ptr,
    x_ptr,
    wg_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for row in range(0, M):
        g = tl.load(og_ptr + row.to(tl.int64) * N + cols, mask=col_mask, other=0.0)
        xv = tl.load(x_ptr + row)
        acc += g.to(tl.float32) * xv.to(tl.float32)
    tl.store(wg_ptr + cols, acc.to(wg_ptr.type.element_ty), mask=col_mask)


class Outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight):
        logger.debug("GEMS OUTER")
        assert inp.ndim == 1 and weight.ndim == 1, "Invalid input"
        if not inp.is_contiguous():
            inp = inp.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        ctx.save_for_backward(inp, weight)
        M = inp.shape[0]
        N = weight.shape[0]
        out = torch.empty((M, N), device=inp.device, dtype=inp.dtype)
        if out.numel() == 0:
            return out
        BLOCK_M, BLOCK_N, num_warps, even, use_i32 = _fwd_blocks(
            M, N, inp.element_size()
        )
        grid = (triton.cdiv(M, BLOCK_M),)
        with torch_device_fn.device(inp.device):
            outer_fwd_kernel[grid](
                inp,
                weight,
                out,
                M,
                N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                EVEN=even,
                I32=use_i32,
                num_warps=num_warps,
            )
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS OUTER VJP")
        assert out_grad.ndim == 2, "invalide out_grad shape"

        inp, weight = ctx.saved_tensors
        if not out_grad.is_contiguous():
            out_grad = out_grad.contiguous()
        M = inp.shape[0]
        N = weight.shape[0]
        inp_grad = torch.empty((M,), device=out_grad.device, dtype=inp.dtype)
        weight_grad = torch.empty((N,), device=out_grad.device, dtype=weight.dtype)
        if M == 0 or N == 0:
            return inp_grad, weight_grad

        BLOCK_M = min(triton.cdiv(M, _FWD_TARGET_CTAS), _FWD_BLOCK_M_CAP)
        BLOCK_M = max(BLOCK_M, triton.cdiv(M, _GRID_X_MAX), 1)
        grid = (triton.cdiv(M, BLOCK_M),)
        with torch_device_fn.device(out_grad.device):
            outer_bwd_inp_kernel[grid](
                out_grad,
                weight,
                inp_grad,
                M,
                N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=_BWD_BLOCK_N,
            )

        BLOCK_N = _BWD_BLOCK_N
        if triton.cdiv(N, BLOCK_N) > _GRID_X_MAX:
            BLOCK_N = triton.next_power_of_2(triton.cdiv(N, _GRID_X_MAX))
        grid = (triton.cdiv(N, BLOCK_N),)
        with torch_device_fn.device(out_grad.device):
            outer_bwd_weight_kernel[grid](
                out_grad, inp, weight_grad, M, N, BLOCK_N=BLOCK_N
            )
        return inp_grad, weight_grad


def outer(inp, weight):
    return Outer.apply(inp, weight)
