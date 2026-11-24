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

"""TLE (Triton Language Extensions) implementation of mhc_post for Ascend NPU.

This module is re-exported as ``mhc_post`` from ``_ascend.ops.__init__``, so on
Ascend the vendor registry replaces the public ``flaggems_vllm.mhc_post`` with
the AscendC-compatible ``(x, h_res, h_out, h_post)`` signature.  The generic
vLLM-shaped implementation stays available from
``flaggems_vllm.ops.mhc.mhc_post``.

1:1 port of the AscendC operator at ``ops-transformer/mhc/mhc_post``
(``op_kernel/mhc_post.cpp``) written in Triton using the TLE DSA surface
(``triton.experimental.tle.dsa``).

Semantics (both AscendC and this file)::

    output[t, i, d] = sum_j h_res[t, j, i] * x[t, j, d]
                    + h_post[t, i]     * h_out[t, d]

AscendC layout -> TLE mapping
-----------------------------
    ``CopyInHOut``  (DataCopyPad 1x dNum*sizeof(T))
        ``hout_ub = tle.dsa.alloc([BLOCK_D], x_dtype, UB)``  +
        ``tle.dsa.copy(h_out_ptr + off, hout_ub, [tail_d])``.

    ``CopyInX`` with ``USE_PERMANENT_X == 1``
        One ``DataCopyPad`` with ``blockCount = n`` /
        ``srcStride = (D - dNum) * sizeof(T)`` -- the whole
        (n, dNum) x-tile is staged in UB once per d-chunk.
        TLE: a single 2D ``tle.dsa.alloc([4, BLOCK_D], ..., UB)`` +
        ``tle.dsa.copy(src_2d, x_ub, [4, tail_d])``.

    ``ComputeCopyOutAllX``
        ``Cast``        -> ``tle.dsa.to_tensor(x_ub).to(tl.float32)``
        ``Muls``        -> ``out_f32 = hout_f32 * h_post[i]``
        ``Axpy``        -> ``out_f32 += x_f32[j] * h_res[j, i]``
                           (one fused multiply-add in tensor land)
        ``Cast RINT``   -> ``out_f32.to(x_dtype)``
        ``CopyOutTile`` -> ``tle.dsa.copy(out_ub, out_ptr + off, [tail_d])``
                           via ``tle.dsa.to_buffer(out_tile, UB)``.

    Scalar ``h_res`` / ``h_post`` GM reads (``GetValue`` inside the
    AscendC inner loop) become per-token scalar ``tl.load`` hoisted
    outside the D loop -- the AscendC arch35 "permanent-x" fast path
    keeps x in UB precisely so these 20 scalars amortize across all
    d-chunks; we keep the same data flow.

Grid
----
The shipped fast path is ``_mhc_post_kernel_tle_persistent``: a *persistent*
launch of ``(vector_core_num,)`` programs, each walking
``for tok in tle.dsa.pipeline(pid, T, npg, ...)`` so one core consumes T/npg
tokens, mirroring ``tilingData_->usedCoreNum`` / ``normalCoreProcessNum`` in the
AscendC tiling.  Measured 1.25-1.31x faster than the older per-token launches
(benchmarks and analysis in ``mhc_post_perf/``).

The older kernels keep the ``(T, cdiv(D, BLOCK_D))`` / ``(T,)`` per-token
distribution and stay reachable through ``use_persistent=False``.
"""

from __future__ import annotations

import functools
import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.mhc.mhc_post import MixLayout

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle as tle

    _HAS_DSA = hasattr(tle, "dsa") and hasattr(tle.dsa, "alloc")
except (ImportError, AttributeError):
    tle = None
    _HAS_DSA = False


# ===========================================================================
# HC_MULT=4 kernel -- maps MhcPostKernel<..., USE_PERMANENT_X=1>::Process.
# Pipeline version: 1D grid over T, iterates D-chunks internally with
# tle.dsa.pipeline(num_stages=2) for MTE2/Vector overlap via double-buffer.
# Uses tle.dsa.to_buffer + tle.dsa.copy for output (matching AscendC CopyOut).
# ===========================================================================
@triton.jit
def _mhc_post_kernel_tle(
    x_ptr,  # (T, 4, D) bf16/fp16
    h_res_ptr,  # (T, 4, 4) fp32   layout: h_res[t, j, i]
    h_out_ptr,  # (T, D)    bf16/fp16
    h_post_ptr,  # (T, 4)    fp32
    out_ptr,  # (T, 4, D) bf16/fp16
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    pid_t = tl.program_id(0)

    x_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    x_dt = x_ptr.dtype.element_ty

    # ---- Bulk DMA: load coefficients (constant across D-chunks) ---------------
    hpost_ub = tle.dsa.alloc([4], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    hres_ub = tle.dsa.alloc([16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(h_post_ptr + hpost_base + tl.arange(0, 4), hpost_ub, [4])
        tle.dsa.copy(h_res_ptr + hres_base + tl.arange(0, 16), hres_ub, [16])

    # ---- Extract coefficient vectors (reused across all D-chunks) -------------
    hp = tle.dsa.to_tensor(hpost_ub)  # [4] f32

    hr_vec = tle.dsa.to_tensor(hres_ub)  # [16] f32
    hr0 = tl.reshape(tle.dsa.extract_slice(hr_vec, (0,), (4,), (1,)), [4])
    hr1 = tl.reshape(tle.dsa.extract_slice(hr_vec, (4,), (4,), (1,)), [4])
    hr2 = tl.reshape(tle.dsa.extract_slice(hr_vec, (8,), (4,), (1,)), [4])
    hr3 = tl.reshape(tle.dsa.extract_slice(hr_vec, (12,), (4,), (1,)), [4])

    # ---- Allocate double-buffered UB for x[4,BLOCK_D] and h_out[BLOCK_D] ------
    ho_ub = tle.dsa.alloc([BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)
    x_ub = tle.dsa.alloc([4, BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)

    # ---- Software-pipelined D-chunk loop --------------------------------------
    for d_chunk in tle.dsa.pipeline(0, NUM_D_BLOCKS, 1, num_stages=2):
        d_start = d_chunk * BLOCK_D
        d_off = d_start + tl.arange(0, BLOCK_D)
        tail_d = tl.minimum(BLOCK_D, D - d_start)

        # -- DMA stage: copy x[4,BLOCK_D] and h_out[BLOCK_D] from GM to UB ---
        n_idx = tl.arange(0, 4)
        src_2d = x_ptr + x_base + n_idx[:, None] * D + d_off[None, :]
        tle.dsa.copy(src_2d, x_ub, [4, tail_d])
        tle.dsa.copy(h_out_ptr + hout_base + d_off, ho_ub, [tail_d])

        # -- Compute stage: parallel FMA across 4 heads -----------------------
        ho = tle.dsa.to_tensor(ho_ub).to(tl.float32)  # [BLOCK_D]
        x2d = tle.dsa.to_tensor(x_ub).to(tl.float32)  # [4, BLOCK_D]
        x0 = tl.reshape(
            tle.dsa.extract_slice(x2d, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x1 = tl.reshape(
            tle.dsa.extract_slice(x2d, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x2 = tl.reshape(
            tle.dsa.extract_slice(x2d, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x3 = tl.reshape(
            tle.dsa.extract_slice(x2d, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        with tle.dsa.hint(inter_no_alias=True):
            Y = tl.expand_dims(hp, 1) * tl.expand_dims(ho, 0)  # [4, BLOCK_D]
            Y += tl.expand_dims(hr0, 1) * tl.expand_dims(x0, 0)
            Y += tl.expand_dims(hr1, 1) * tl.expand_dims(x1, 0)
            Y += tl.expand_dims(hr2, 1) * tl.expand_dims(x2, 0)
            Y += tl.expand_dims(hr3, 1) * tl.expand_dims(x3, 0)

        # -- Store: to_buffer + dsa.copy (matching AscendC CopyOutTile) -------
        y0 = tl.reshape(
            tle.dsa.extract_slice(Y, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y1 = tl.reshape(
            tle.dsa.extract_slice(Y, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y2 = tl.reshape(
            tle.dsa.extract_slice(Y, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y3 = tl.reshape(
            tle.dsa.extract_slice(Y, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        y0b = tle.dsa.to_buffer(y0.to(x_dt), tle.dsa.ascend.UB)
        y1b = tle.dsa.to_buffer(y1.to(x_dt), tle.dsa.ascend.UB)
        y2b = tle.dsa.to_buffer(y2.to(x_dt), tle.dsa.ascend.UB)
        y3b = tle.dsa.to_buffer(y3.to(x_dt), tle.dsa.ascend.UB)
        with tle.dsa.hint(inter_no_alias=True):
            tle.dsa.copy(y0b, out_ptr + x_base + 0 * D + d_off, [tail_d])
            tle.dsa.copy(y1b, out_ptr + x_base + 1 * D + d_off, [tail_d])
            tle.dsa.copy(y2b, out_ptr + x_base + 2 * D + d_off, [tail_d])
            tle.dsa.copy(y3b, out_ptr + x_base + 3 * D + d_off, [tail_d])


# ===========================================================================
# Row-major-safe variant: builds the (4, BLOCK_D) output tile with
# tl.join on the *leading* axis so memory order matches (n, D).
# Kept separate so the compiler only sees one layout path per kernel.
# ===========================================================================
@triton.jit
def _mhc_post_kernel_tle_rows(
    x_ptr,  # (T, 4, D) bf16/fp16
    h_res_ptr,  # (T, 4, 4) fp32
    h_out_ptr,  # (T, D)    bf16/fp16
    h_post_ptr,  # (T, 4)    fp32
    out_ptr,  # (T, 4, D) bf16/fp16
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    x_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    x_dt = x_ptr.dtype.element_ty

    # ---- Bulk DMA: merge all tl.load into large tensor DSA copies -----------
    # h_post[t, :] -> UB [4] fp32 (exact size, no tail issue)
    hpost_ub = tle.dsa.alloc([4], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    # h_res[t, :, :] -> UB [16] fp32 (exact size, no tail issue)
    hres_ub = tle.dsa.alloc([16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(h_post_ptr + hpost_base + tl.arange(0, 4), hpost_ub, [4])
        tle.dsa.copy(h_res_ptr + hres_base + tl.arange(0, 16), hres_ub, [16])

    # ---- Extract coefficient vectors from UB ----------------------------------
    # h_post: [4] f32 — broadcast coeff for h_out
    hp = tle.dsa.to_tensor(hpost_ub)  # [4] f32

    # h_res flat [16] layout: [r00,r01,r02,r03, r10,r11,r12,r13, r20,..., r30,...]
    # Row j of h_res[j,i] contains the coefficients of x[j] for all 4 output heads.
    # Extract contiguous [4] slices — one per input x row:
    hr_vec = tle.dsa.to_tensor(hres_ub)  # [16] f32
    hr0 = tl.reshape(
        tle.dsa.extract_slice(hr_vec, (0,), (4,), (1,)), [4]
    )  # [r00,r01,r02,r03]
    hr1 = tl.reshape(
        tle.dsa.extract_slice(hr_vec, (4,), (4,), (1,)), [4]
    )  # [r10,r11,r12,r13]
    hr2 = tl.reshape(
        tle.dsa.extract_slice(hr_vec, (8,), (4,), (1,)), [4]
    )  # [r20,r21,r22,r23]
    hr3 = tl.reshape(
        tle.dsa.extract_slice(hr_vec, (12,), (4,), (1,)), [4]
    )  # [r30,r31,r32,r33]

    # ---- Load x, h_out with mask (handles non-power-of-2 D correctly) ---------
    ho = tl.load(h_out_ptr + hout_base + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + x_base + 0 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + x_base + 1 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + x_base + 2 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + x_base + 3 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)

    # ---- Parallel vector multiply + accumulate --------------------------------
    # Broadcast coeff[4,1] * data[1,BLOCK_D] → [4,BLOCK_D], accumulate 5 terms.
    # Each term computes contributions of one input vector to ALL 4 output heads
    # simultaneously, instead of computing each head independently.
    #   Y[i, d] = hp[i] * ho[d] + hr0[i] * x0[d] + hr1[i] * x1[d]
    #           + hr2[i] * x2[d] + hr3[i] * x3[d]
    Y = tl.expand_dims(hp, 1) * tl.expand_dims(ho, 0)  # [4, BLOCK_D]
    Y += tl.expand_dims(hr0, 1) * tl.expand_dims(x0, 0)  # + [4, BLOCK_D]
    Y += tl.expand_dims(hr1, 1) * tl.expand_dims(x1, 0)  # + [4, BLOCK_D]
    Y += tl.expand_dims(hr2, 1) * tl.expand_dims(x2, 0)  # + [4, BLOCK_D]
    Y += tl.expand_dims(hr3, 1) * tl.expand_dims(x3, 0)  # + [4, BLOCK_D]

    # Extract output rows
    y0 = tl.reshape(tle.dsa.extract_slice(Y, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])
    y1 = tl.reshape(tle.dsa.extract_slice(Y, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])
    y2 = tl.reshape(tle.dsa.extract_slice(Y, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])
    y3 = tl.reshape(tle.dsa.extract_slice(Y, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])

    # ---- Cast f32 -> T + store with mask ---------------------------------------
    tl.store(out_ptr + x_base + 0 * D + d_off, y0.to(x_dt), mask=d_mask)
    tl.store(out_ptr + x_base + 1 * D + d_off, y1.to(x_dt), mask=d_mask)
    tl.store(out_ptr + x_base + 2 * D + d_off, y2.to(x_dt), mask=d_mask)
    tl.store(out_ptr + x_base + 3 * D + d_off, y3.to(x_dt), mask=d_mask)


# ===========================================================================
# Pipeline variant: iterates over D-chunks inside the kernel with
# tle.dsa.pipeline for software-pipelined double-buffer overlap between
# MTE2 (load) and Vector (compute).  Grid is 1D: (T,).
# ===========================================================================
@triton.jit
def _mhc_post_kernel_tle_rows_pipeline(
    x_ptr,  # (T, 4, D) bf16/fp16
    h_res_ptr,  # (T, 4, 4) fp32
    h_out_ptr,  # (T, D)    bf16/fp16
    h_post_ptr,  # (T, 4)    fp32
    out_ptr,  # (T, 4, D) bf16/fp16
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    pid_t = tl.program_id(0)

    x_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    x_dt = x_ptr.dtype.element_ty

    # ---- Bulk DMA: load coefficients (constant across D-chunks) ---------------
    # h_post[t, :] -> UB [4] fp32 (exact size, no tail issue)
    hpost_ub = tle.dsa.alloc([4], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    # h_res[t, :, :] -> UB [16] fp32 (exact size, no tail issue)
    hres_ub = tle.dsa.alloc([16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(h_post_ptr + hpost_base + tl.arange(0, 4), hpost_ub, [4])
        tle.dsa.copy(h_res_ptr + hres_base + tl.arange(0, 16), hres_ub, [16])

    # ---- Extract coefficient vectors (reused across all D-chunks) -------------
    hp = tle.dsa.to_tensor(hpost_ub)  # [4] f32

    hr_vec = tle.dsa.to_tensor(hres_ub)  # [16] f32
    hr0 = tl.reshape(tle.dsa.extract_slice(hr_vec, (0,), (4,), (1,)), [4])
    hr1 = tl.reshape(tle.dsa.extract_slice(hr_vec, (4,), (4,), (1,)), [4])
    hr2 = tl.reshape(tle.dsa.extract_slice(hr_vec, (8,), (4,), (1,)), [4])
    hr3 = tl.reshape(tle.dsa.extract_slice(hr_vec, (12,), (4,), (1,)), [4])

    # ---- Allocate double-buffered UB for x[4,BLOCK_D] and h_out[BLOCK_D] ------
    # tle.dsa.pipeline with num_stages=2 enables MTE2/V overlap via double buffer.
    ho_ub = tle.dsa.alloc([BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)
    x_ub = tle.dsa.alloc([4, BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)

    # ---- Software-pipelined D-chunk loop --------------------------------------
    # tle.dsa.pipeline(start, end, step, num_stages=2) generates double-buffered
    # DMA/compute overlap: while computing on chunk N, loads chunk N+1.
    for d_chunk in tle.dsa.pipeline(0, NUM_D_BLOCKS, 1, num_stages=2):
        d_start = d_chunk * BLOCK_D
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D
        tail_d = tl.minimum(BLOCK_D, D - d_start)

        # -- DMA stage: copy x[4,BLOCK_D] and h_out[BLOCK_D] from GM to UB ---
        n_idx = tl.arange(0, 4)
        src_2d = x_ptr + x_base + n_idx[:, None] * D + d_off[None, :]
        tle.dsa.copy(src_2d, x_ub, [4, tail_d])
        tle.dsa.copy(h_out_ptr + hout_base + d_off, ho_ub, [tail_d])

        # -- Compute stage: parallel FMA across 4 heads -----------------------
        ho = tle.dsa.to_tensor(ho_ub).to(tl.float32)  # [BLOCK_D]
        x2d = tle.dsa.to_tensor(x_ub).to(tl.float32)  # [4, BLOCK_D]
        x0 = tl.reshape(
            tle.dsa.extract_slice(x2d, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x1 = tl.reshape(
            tle.dsa.extract_slice(x2d, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x2 = tl.reshape(
            tle.dsa.extract_slice(x2d, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x3 = tl.reshape(
            tle.dsa.extract_slice(x2d, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        # Compute all 5 terms separately, then sum once
        term0 = tl.expand_dims(hp, 1) * tl.expand_dims(ho, 0)  # [4, BLOCK_D]
        term1 = tl.expand_dims(hr0, 1) * tl.expand_dims(x0, 0)  # [4, BLOCK_D]
        term2 = tl.expand_dims(hr1, 1) * tl.expand_dims(x1, 0)  # [4, BLOCK_D]
        term3 = tl.expand_dims(hr2, 1) * tl.expand_dims(x2, 0)  # [4, BLOCK_D]
        term4 = tl.expand_dims(hr3, 1) * tl.expand_dims(x3, 0)  # [4, BLOCK_D]
        Y = term0 + term1 + term2 + term3 + term4  # [4, BLOCK_D]

        # -- Extract rows and store back to GM --------------------------------
        y0 = tl.reshape(
            tle.dsa.extract_slice(Y, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y1 = tl.reshape(
            tle.dsa.extract_slice(Y, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y2 = tl.reshape(
            tle.dsa.extract_slice(Y, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y3 = tl.reshape(
            tle.dsa.extract_slice(Y, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        tl.store(out_ptr + x_base + 0 * D + d_off, y0.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 1 * D + d_off, y1.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 2 * D + d_off, y2.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 3 * D + d_off, y3.to(x_dt), mask=d_mask)


# ===========================================================================
# Concat + reduce variant: stack terms and use explicit summation.
# ===========================================================================
@triton.jit
def _mhc_post_kernel_tle_concat_reduce(
    x_ptr,  # (T, 4, D) bf16/fp16
    h_res_ptr,  # (T, 4, 4) fp32
    h_out_ptr,  # (T, D)    bf16/fp16
    h_post_ptr,  # (T, 4)    fp32
    out_ptr,  # (T, 4, D) bf16/fp16
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    """Alternative implementation that computes all 5 terms separately then sums.

    Compute in FP32 for intermediate results, cast to output dtype once at the end.
    This matches the original pipeline version's precision but with better scheduling.
    """
    pid_t = tl.program_id(0)

    x_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    x_dt = x_ptr.dtype.element_ty

    # ---- Bulk DMA: load coefficients (constant across D-chunks) ---------------
    hpost_ub = tle.dsa.alloc([4], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    hres_ub = tle.dsa.alloc([16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(h_post_ptr + hpost_base + tl.arange(0, 4), hpost_ub, [4])
        tle.dsa.copy(h_res_ptr + hres_base + tl.arange(0, 16), hres_ub, [16])

    # ---- Extract coefficient vectors (keep in FP32) --------------------------
    hp = tle.dsa.to_tensor(hpost_ub)  # [4] f32
    hr_vec = tle.dsa.to_tensor(hres_ub)  # [16] f32
    hr0 = tl.reshape(tle.dsa.extract_slice(hr_vec, (0,), (4,), (1,)), [4])
    hr1 = tl.reshape(tle.dsa.extract_slice(hr_vec, (4,), (4,), (1,)), [4])
    hr2 = tl.reshape(tle.dsa.extract_slice(hr_vec, (8,), (4,), (1,)), [4])
    hr3 = tl.reshape(tle.dsa.extract_slice(hr_vec, (12,), (4,), (1,)), [4])

    # ---- Allocate double-buffered UB ----------------------------------------
    ho_ub = tle.dsa.alloc([BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)
    x_ub = tle.dsa.alloc([4, BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)

    # ---- Software-pipelined D-chunk loop --------------------------------------
    for d_chunk in tle.dsa.pipeline(0, NUM_D_BLOCKS, 1, num_stages=2):
        d_start = d_chunk * BLOCK_D
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D
        tail_d = tl.minimum(BLOCK_D, D - d_start)

        # -- DMA stage: load data ---
        n_idx = tl.arange(0, 4)
        src_2d = x_ptr + x_base + n_idx[:, None] * D + d_off[None, :]
        tle.dsa.copy(src_2d, x_ub, [4, tail_d])
        tle.dsa.copy(h_out_ptr + hout_base + d_off, ho_ub, [tail_d])

        # -- Compute: compute 5 separate outer products [4, BLOCK_D] each in FP32 ---
        ho = tle.dsa.to_tensor(ho_ub).to(tl.float32)  # [BLOCK_D]
        x2d = tle.dsa.to_tensor(x_ub).to(tl.float32)  # [4, BLOCK_D]
        x0 = tl.reshape(
            tle.dsa.extract_slice(x2d, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x1 = tl.reshape(
            tle.dsa.extract_slice(x2d, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x2 = tl.reshape(
            tle.dsa.extract_slice(x2d, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x3 = tl.reshape(
            tle.dsa.extract_slice(x2d, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        # Compute 5 terms, each is [4, BLOCK_D] in FP32
        term0 = tl.expand_dims(hp, 1) * tl.expand_dims(ho, 0)  # [4, BLOCK_D]
        term1 = tl.expand_dims(hr0, 1) * tl.expand_dims(x0, 0)  # [4, BLOCK_D]
        term2 = tl.expand_dims(hr1, 1) * tl.expand_dims(x1, 0)  # [4, BLOCK_D]
        term3 = tl.expand_dims(hr2, 1) * tl.expand_dims(x2, 0)  # [4, BLOCK_D]
        term4 = tl.expand_dims(hr3, 1) * tl.expand_dims(x3, 0)  # [4, BLOCK_D]

        # Sum all terms in FP32
        Y = term0 + term1 + term2 + term3 + term4  # [4, BLOCK_D]

        # -- Extract rows, cast once, and store ---
        y0 = tl.reshape(
            tle.dsa.extract_slice(Y, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y1 = tl.reshape(
            tle.dsa.extract_slice(Y, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y2 = tl.reshape(
            tle.dsa.extract_slice(Y, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y3 = tl.reshape(
            tle.dsa.extract_slice(Y, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        tl.store(out_ptr + x_base + 0 * D + d_off, y0.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 1 * D + d_off, y1.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 2 * D + d_off, y2.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 3 * D + d_off, y3.to(x_dt), mask=d_mask)


# ===========================================================================
# Persistent grid-stride kernel: the AscendC data flow, expressed in Triton.
#
# This is the fast path for hc_mult=4.  It differs from the ``rows`` / ``rows_
# pipeline`` kernels above in three ways that were each measured on a 910B4
# (T=8192, D=3584, bf16, bytes = 529 MB):
#
# 1. *Persistent launch* -- ``grid=(num_vector_cores,)`` and each program walks
#    ``for tok in tle.dsa.pipeline(pid, T, npg, num_stages=2)``.  The old 1D
#    ``grid=(T,)`` launch gave one program per token, so the D-chunk pipeline
#    loop had exactly one iteration (``NUM_D_BLOCKS == 1``) and no DMA/vector
#    overlap could ever be expressed.  Per-program block setup is also amortized
#    over T/npg tokens.
#
# 2. *Row-wise FMA math* -- ``Muls`` + 4 x ``Axpy``, i.e. 1 multiply + 4 fused
#    multiply-adds per output element, matching ``MhcPostKernel::
#    ComputeCopyOutAllX``.  The older kernels materialize five ``[4, BLOCK_D]``
#    fp32 term tensors and fold them with four full-tile adds (9 vector ops per
#    output element, plus a 64 KB temp).  Measured: 885us -> 703us.
#
#    Known remaining gap: the last step of that match wants the *fused*
#    accumulate, which Triton cannot express.  ``acc + coeff * x`` lowers to
#    ``vmuls`` + ``vadd``, and arch22 has no ``fma`` mnemonic to pattern-match
#    against (the fused forms are ``vaxpy`` / ``vmadd`` / ``vmla``).  So this
#    kernel spends nine vector instructions per output element where AscendC
#    spends five.  Measured on a 910B4: a hand-written AscendC custom op
#    emitting ``vaxpy`` takes T=8192 from 698us to 473us (1.48x, bit-identical
#    output), which is what the fused instruction is worth here.  Closing it
#    belongs in the compiler (pattern-match the add-of-multiply), not in a
#    per-operator custom op, so this kernel stays on ``tl.fma``.
#
# 3. *No lane padding* -- ``BLOCK_D`` divides ``D`` (3584) instead of being the
#    next power of two (4096), so the vector unit does not spin on 12.5% dead
#    lanes.  Measured: a further 880us -> 821us at T=8192.
#
# Coefficients are read as scalars (``tl.load`` of one element), which is the
# same access the AscendC kernel performs with ``hResGm_.GetValue``.
#
# NOTE on overlap: ``tle.dsa.copy`` lowers to a blocking GM->UB DMA in this
# toolchain, so the DMA and vector phases still serialize (measured: the whole
# TLE pipeline body is ~= memory time + vector time).  ``num_stages``,
# ``multibuffer=True`` and a hand written ping-pong over two UB buffer sets were
# all tried and none of them changed the result by more than 1%, so the
# remaining gap to the AscendC kernel is the DMA/vector overlap that AscendC
# gets from its depth-2 ``TQue``.
# ===========================================================================
@triton.jit
def _mhc_post_store_rows(
    ho,
    x0,
    x1,
    x2,
    x3,
    h_res_ptr,
    h_post_ptr,
    out_ptr,
    xb,
    off,
    tok,
    D: tl.constexpr,
    TAIL: tl.constexpr,
):
    """Emit the 4 output rows of one token/one d-chunk.

    ``out[i, d] = h_post[t, i] * h_out[d] + sum_j h_res[t, j, i] * x[j, d]``,
    evaluated as one multiply plus four FMAs per output element.
    """
    x_dt = out_ptr.dtype.element_ty
    hr_b = tok * 16
    hp_b = tok * 4
    for i in tl.static_range(0, 4):
        acc = tl.load(h_post_ptr + hp_b + i) * ho
        acc = tl.fma(tl.load(h_res_ptr + hr_b + i), x0, acc)
        acc = tl.fma(tl.load(h_res_ptr + hr_b + 4 + i), x1, acc)
        acc = tl.fma(tl.load(h_res_ptr + hr_b + 8 + i), x2, acc)
        acc = tl.fma(tl.load(h_res_ptr + hr_b + 12 + i), x3, acc)
        tle.dsa.copy(
            tle.dsa.to_buffer(acc.to(x_dt), tle.dsa.ascend.UB),
            out_ptr + xb + i * D + off,
            [TAIL],
            inter_no_alias=True,
        )


@triton.jit
def _mhc_post_kernel_tle_persistent(
    x_ptr,  # (T, 4, D) bf16/fp16
    h_res_ptr,  # (T, 4, 4) fp32   layout: h_res[t, j, i]
    h_out_ptr,  # (T, D)    bf16/fp16
    h_post_ptr,  # (T, 4)    fp32
    out_ptr,  # (T, 4, D) bf16/fp16
    T,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    x_dt = x_ptr.dtype.element_ty
    off = tl.arange(0, BLOCK_D)

    for tok in tle.dsa.pipeline(pid, T, npg, num_stages=2):
        xb = tok * 4 * D
        for d in tl.static_range(0, NUM_D_BLOCKS):
            d_start = d * BLOCK_D
            d_off = d_start + off
            tail = tl.minimum(BLOCK_D, D - d_start)

            ho_ub = tle.dsa.alloc([BLOCK_D], x_dt, tle.dsa.ascend.UB)
            x0_ub = tle.dsa.alloc([BLOCK_D], x_dt, tle.dsa.ascend.UB)
            x1_ub = tle.dsa.alloc([BLOCK_D], x_dt, tle.dsa.ascend.UB)
            x2_ub = tle.dsa.alloc([BLOCK_D], x_dt, tle.dsa.ascend.UB)
            x3_ub = tle.dsa.alloc([BLOCK_D], x_dt, tle.dsa.ascend.UB)

            tle.dsa.copy(
                h_out_ptr + tok * D + d_off, ho_ub, [tail], inter_no_alias=True
            )
            tle.dsa.copy(x_ptr + xb + 0 * D + d_off, x0_ub, [tail], inter_no_alias=True)
            tle.dsa.copy(x_ptr + xb + 1 * D + d_off, x1_ub, [tail], inter_no_alias=True)
            tle.dsa.copy(x_ptr + xb + 2 * D + d_off, x2_ub, [tail], inter_no_alias=True)
            tle.dsa.copy(x_ptr + xb + 3 * D + d_off, x3_ub, [tail], inter_no_alias=True)

            ho = tle.dsa.to_tensor(ho_ub, writable=False).to(tl.float32)
            x0 = tle.dsa.to_tensor(x0_ub, writable=False).to(tl.float32)
            x1 = tle.dsa.to_tensor(x1_ub, writable=False).to(tl.float32)
            x2 = tle.dsa.to_tensor(x2_ub, writable=False).to(tl.float32)
            x3 = tle.dsa.to_tensor(x3_ub, writable=False).to(tl.float32)

            _mhc_post_store_rows(
                ho,
                x0,
                x1,
                x2,
                x3,
                h_res_ptr,
                h_post_ptr,
                out_ptr,
                xb,
                d_off,
                tok,
                D=D,
                TAIL=tail,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _contig(t):
    """``contiguous()`` only when it is not already a no-op.

    ``Tensor.contiguous()`` on an already-contiguous tensor is free on device
    but still costs a dispatcher round trip; the op is host-bound for small
    token counts, so the check is worth it.
    """
    return t if t.is_contiguous() else t.contiguous()


def _flatten_bsn(x, h_res, h_out, h_post):
    """Return (x, h_res, h_out, h_post, restore_shape) in (T, n, D) layout.

    For the common (T, n, D) call the arguments are already in the wanted
    layout, and the operator is host-bound for small token counts, so the
    reshape/contiguous no-ops are skipped by a shape check instead of being
    dispatched (``_flatten_bsn`` alone was 14us of a ~170us call).
    """
    if x.dim() == 4:
        B, S, N, D = x.shape
        T = B * S
        x = _contig(x.reshape(T, N, D))
        h_res = _contig(h_res.reshape(T, N, N))
        h_out = _contig(h_out.reshape(T, D))
        h_post = _contig(h_post.reshape(T, N))
        return x, h_res, h_out, h_post, (B, S, N, D)
    if x.dim() == 3:
        T, N, D = x.shape
        # ``h_post`` may arrive as (T, N) or (T, N, 1): the public wrapper takes
        # post_layer_mix in the vLLM layout, where the trailing dimension is 1.
        # Both are contiguous with identical element layout and the kernel only
        # reads the base pointer, so neither is reshaped here -- that saved a
        # ~7us dispatcher round trip per call.
        hp_shape = tuple(h_post.shape)
        if (
            x.is_contiguous()
            and tuple(h_res.shape) == (T, N, N)
            and h_res.is_contiguous()
            and tuple(h_out.shape) == (T, D)
            and h_out.is_contiguous()
            and (hp_shape == (T, N) or hp_shape == (T, N, 1))
            and h_post.is_contiguous()
        ):
            return x, h_res, h_out, h_post, (T, N, D)
        return (
            _contig(x),
            _contig(h_res.reshape(T, N, N)),
            _contig(h_out.reshape(T, D)),
            _contig(h_post.reshape(T, N)),
            (T, N, D),
        )
    raise ValueError(f"unsupported x.dim()={x.dim()}, expect 3 or 4")


_AIV_CORES = None


def _aiv_core_count() -> int:
    """Number of vector (AIV) cores to launch the persistent kernel with.

    The persistent grid must match the physical vector-core count: too few
    leaves cores idle (measured T=8192: grid=32 -> 858us vs grid=40 -> 695us),
    too many makes the surplus blocks queue behind the first wave (grid=80 ->
    701us on a 40-core part).

    Cached: ``get_device_properties`` is a host round trip, and this operator is
    host-bound for small token counts (~+45us/call if not cached).
    """
    global _AIV_CORES
    if _AIV_CORES is not None:
        return _AIV_CORES
    count = 0
    try:
        props = torch.npu.get_device_properties(torch.npu.current_device())
        for attr in ("vector_core_num", "multi_processor_count"):
            num = getattr(props, attr, 0) or 0
            if num > 0:
                count = int(num)
                break
    except Exception:  # noqa: BLE001 - fall back to the Atlas A2 1-die width
        count = 0
    _AIV_CORES = count or 40
    return _AIV_CORES


@functools.lru_cache(maxsize=None)
def _pick_block_d(D: int):
    """Largest d-chunk that divides ``D`` and still fits a UB tile.

    The old code used ``next_power_of_2(D)``, which pads D=3584 up to 4096 and
    burns 12.5% of the vector lanes.  Chunking must also stay as coarse as
    possible: every extra chunk re-issues the DMA setup and re-walks the loop,
    and measurement showed 1 chunk (821us) beats 2 (967us) and 4 (1052us) at
    T=8192.  Returns ``(BLOCK_D, NUM_D_BLOCKS)`` or ``None`` when no clean
    divisor exists (caller then keeps the padded split).
    """
    for k in (1, 2, 4, 8):
        if D % k:
            continue
        bd = D // k
        if bd <= 4096 and bd % 16 == 0:
            return bd, k
    return None


@functools.lru_cache(maxsize=None)
def _padded_tiling(D: int, T: int):
    """``(BLOCK_D, NUM_D_BLOCKS)`` for the padded (non-persistent) kernels.

    ``BLOCK_D`` starts at the next power of two so the vector unit can mask off
    the tail, then doubles until ``T * NUM_D_BLOCKS`` fits the 65535 coreDim
    cap.  Cached: the loop is pure arithmetic but ``triton.next_power_of_2``
    and ``triton.cdiv`` are Python calls that cost ~7us together per
    invocation, which is 4% of a T=1024 call, and the result only depends on
    ``(D, T)``.
    """
    npo2 = triton.next_power_of_2(D)
    block_d = min(4096, npo2)
    while T * triton.cdiv(D, block_d) > 65535 and block_d < D:
        block_d *= 2
    block_d = min(block_d, npo2)
    return block_d, triton.cdiv(D, block_d)


# Triton's ``kernel[grid](...)`` redoes argument binding, key computation and
# cache lookup on every call; measured 81.7us at T=1024, i.e. more than the
# kernel itself (59us).  Once a kernel has been specialised, replaying it
# through the ``CompiledKernel`` that the first launch returned costs 59.3us,
# so the launch is worth caching.  Keys carry everything that affects
# specialisation -- constexpr values, dtypes, and the pointer alignment bits
# Triton also keys on -- so a hit is always a binary built for these arguments.
_LAUNCHERS: dict = {}


def _ptr_alignment_bits(*tensors) -> int:
    """Bitmask of which arguments are 16-byte aligned.

    Triton specialises kernels on that property, so it has to be part of the
    cache key: replaying a binary specialised for an aligned pointer with an
    unaligned one would run the wrong code.
    """
    bits = 0
    for i, t in enumerate(tensors):
        if t is not None and (t.data_ptr() & 15) == 0:
            bits |= 1 << i
    return bits


def _launch_cached(jit_kernel, grid, key, args):
    """Launch ``jit_kernel`` through its CompiledKernel, compiling on first use.

    Falls back to Triton's normal dispatch if the cached replay raises, so a
    launcher ABI change degrades to the slow-but-correct path instead of
    breaking the operator.
    """
    compiled = _LAUNCHERS.get(key)
    if compiled is None:
        _LAUNCHERS[key] = jit_kernel[tuple(grid)](*args)
        return
    try:
        padded = tuple(grid) + (1,) * (3 - len(grid))
        compiled[padded](*args)
    except Exception:  # noqa: BLE001 - a fast path must never break the op
        _LAUNCHERS.pop(key, None)
        jit_kernel[tuple(grid)](*args)


def _mhc_post_ascend(
    x: torch.Tensor,
    h_res: torch.Tensor,
    h_out: torch.Tensor,
    h_post: torch.Tensor,
    use_pipeline: bool = True,
    use_concat_reduce: bool = False,
    use_persistent: bool = True,
) -> torch.Tensor:
    """TLE kernel entry point using the AscendC ``(x, h_res, h_out, h_post)``
    signature.  ``flaggems_vllm.mhc_post`` wraps this with the generic vLLM
    signature; see :func:`mhc_post`."""
    if not _HAS_DSA:
        raise RuntimeError(
            "This mhc_post implementation requires the TLE DSA surface "
            "(triton.experimental.tle.dsa.*)."
        )
    if h_res is None:
        raise NotImplementedError(
            "mhc_post requires h_res; the h_res=None residual shortcut "
            "is not implemented by this TLE port."
        )

    xf, hres, hout, hpost, shape = _flatten_bsn(x, h_res, h_out, h_post)
    T, N, D = xf.shape
    if N != 4:
        raise NotImplementedError(f"mhc_post only supports hc_mult=4, got hc_mult={N}")
    out = torch.empty_like(xf)
    restore = out if shape == (T, N, D) else None

    # AscendC picks dInner from the tiling table; we mirror it with a fixed
    # BLOCK_D that keeps T * cdiv(D, BLOCK_D) under the 65535 coreDim cap.
    #
    # 4096 is the largest d-chunk whose (4, BLOCK_D) fp32 compute tile still
    # fits UB; raising it from 1024 to 4096 quarters the number of d-chunks per
    # token, so each program issues fewer, longer DMAs.  Measured on a 910B4:
    # 4096 tokens x 3584 hidden 566us -> 440us, 8192 x 3584 1115us -> 878us,
    # with every smaller shape unchanged (they are launch-bound).  Whole-row
    # chunks (BLOCK_D >= 8192 for hidden=7168) overflow UB, hence the cap.
    BLOCK_D, NUM_D_BLOCKS = _padded_tiling(D, T)

    # fp32 coefficients are read directly by the kernel; only pay for a cast
    # when the caller actually handed us something else.
    if hres.dtype is not torch.float32:
        hres = hres.to(torch.float32)
    if hpost.dtype is not torch.float32:
        hpost = hpost.to(torch.float32)

    pick = _pick_block_d(D) if use_persistent else None
    if pick is not None:
        # Fast path: persistent grid-stride launch over the vector cores.
        bd, nblocks = pick
        npg = min(T, _aiv_core_count())
        _launch_cached(
            _mhc_post_kernel_tle_persistent,
            (npg,),
            (
                "persistent",
                D,
                bd,
                nblocks,
                # T is part of the key even though it is a runtime argument:
                # Triton also specialises *integer* arguments on divisibility
                # by 16, so a kernel compiled for T=2 is not valid for T=64.
                # Dropping it here left part of the output unwritten for token
                # counts on the other side of a 16-boundary.
                T,
                xf.dtype,
                _ptr_alignment_bits(xf, hres, hout, hpost, out),
            ),
            (xf, hres, hout, hpost, out, T, D, bd, nblocks),
        )
        return restore if restore is not None else out.reshape(shape)

    if use_concat_reduce:
        # Concat+reduce variant: 1D grid over T
        grid = (T,)
        _mhc_post_kernel_tle_concat_reduce[grid](
            xf,
            hres,
            hout,
            hpost,
            out,
            D=D,
            BLOCK_D=BLOCK_D,
            NUM_D_BLOCKS=NUM_D_BLOCKS,
        )
    elif use_pipeline:
        # Pipeline kernel: 1D grid over T, iterates D-chunks internally
        # with tle.dsa.pipeline(num_stages=2) for MTE2/Vector overlap.
        grid = (T,)
        _mhc_post_kernel_tle_rows_pipeline[grid](
            xf,
            hres,
            hout,
            hpost,
            out,
            D=D,
            BLOCK_D=BLOCK_D,
            NUM_D_BLOCKS=NUM_D_BLOCKS,
        )
    else:
        # Non-pipelined: 2D grid (T, cdiv(D, BLOCK_D))
        grid = (T, NUM_D_BLOCKS)
        _mhc_post_kernel_tle_rows[grid](
            xf,
            hres,
            hout,
            hpost,
            out,
            D=D,
            BLOCK_D=BLOCK_D,
        )
    return out.reshape(shape)


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    *,
    mix_layout: MixLayout = MixLayout.TRANSPOSED,
) -> torch.Tensor:
    """Ascend backend ``mhc_post`` with the generic vLLM signature.

    The argument list is identical to
    ``flaggems_vllm.ops.mhc.mhc_post.mhc_post``.  ``hc_mult == 4`` on NPU is
    served by the TLE kernel; every other case (other ``hc_mult``, non-NPU
    tensors, or a Triton build without TLE) is delegated to the generic
    implementation so the public contract does not change.

    Note:
        The AscendC/TLE kernel reads its ``h_res`` operand as a contiguous
        ``(N, hc, hc)`` block.  ``mix_layout=NATURAL`` and any non-contiguous
        ``comb_res_mix`` view therefore materialise one transpose copy here,
        whereas the generic implementation keeps ``comb_res_mix`` zero-copy.
    """
    if not isinstance(mix_layout, MixLayout):
        raise TypeError(
            "mix_layout must be a MixLayout enum member, "
            f"got {type(mix_layout).__name__}"
        )
    if residual.dtype != torch.bfloat16:
        raise NotImplementedError("mHC post residual must use bfloat16")
    if residual.device.type not in ("cuda", "musa", "npu"):
        raise NotImplementedError("mHC post requires CUDA, MUSA or NPU tensors")
    if residual.dim() != 3:
        raise ValueError(
            f"residual must have shape (N, hc, H), got {tuple(residual.shape)}"
        )

    N, hc, H = residual.shape
    if x.shape != (N, H):
        raise ValueError(f"x must have shape ({N}, {H}), got {tuple(x.shape)}")
    if post_layer_mix.shape not in ((N, hc, 1), (N, hc)):
        raise ValueError(
            f"post_layer_mix must have shape ({N}, {hc}, 1), "
            f"got {tuple(post_layer_mix.shape)}"
        )
    if comb_res_mix.shape != (N, hc, hc):
        raise ValueError(
            f"comb_res_mix must have shape ({N}, {hc}, {hc}), "
            f"got {tuple(comb_res_mix.shape)}"
        )

    if hc != 4 or residual.device.type != "npu" or not _HAS_DSA:
        from flaggems_vllm.ops.mhc.mhc_post import mhc_post as _generic_mhc_post

        return _generic_mhc_post(
            x, residual, post_layer_mix, comb_res_mix, mix_layout=mix_layout
        )

    # Map the generic vLLM layout onto the AscendC one:
    #   generic: out[n,i,h] = post[n,i] * x[n,h]
    #                      + sum_j mix[n,i,j] * residual[n,j,h]
    #   TLE:     out[n,i,h] = sum_j h_res[n,j,i] * x_tle[n,j,h]
    #                      + h_post[n,i] * h_out[n,h]
    # with x_tle=residual, h_out=x, h_post=post_layer_mix.
    h_res = comb_res_mix
    if mix_layout is MixLayout.NATURAL:
        h_res = h_res.transpose(-1, -2)
    if not h_res.is_contiguous():
        h_res = h_res.contiguous()
    # post_layer_mix is (N, hc, 1) or (N, hc); both are contiguous with the same
    # element layout, and _flatten_bsn accepts either, so it is passed straight
    # through instead of paying a reshape dispatch here.
    return _mhc_post_ascend(residual, h_res, x, post_layer_mix)


__all__ = [
    "mhc_post",
    "_mhc_post_ascend",
    "_mhc_post_kernel_tle",
    "_mhc_post_kernel_tle_rows",
    "_mhc_post_kernel_tle_rows_pipeline",
    "_mhc_post_kernel_tle_persistent",
    "_mhc_post_kernel_tle_concat_reduce",
]
