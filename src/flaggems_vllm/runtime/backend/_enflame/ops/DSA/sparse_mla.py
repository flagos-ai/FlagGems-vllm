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

# GCU adaptation of ``DSA/sparse_mla``.
#
# The generic implementation is a tl.dot attention kernel with a 3-D grid
# (B, SQ, VG*NH). On the GCU it fails three independent ways:
#   1. fresh compiles of any tl.dot / 2-D-tile kernel hit the
#      ``memref_ext.slice_start`` / ``gcu.matmul`` backend regression
#      introduced by the 2026-11-07 20:22 toolchain update (see
#      compiler_issues/vllm_parallel_nsa_gcu_matmul.md), so neither the
#      TLE nor the portable kernel can even be built;
#   2. with pre-regression cached binaries the BH=64 specialization
#      exceeded the gcu400 DSM budget (Required: 1122304 > 917504);
#   3. grid.y = SQ violates the GCU grid.y hardware limit (255) for
#      seq_len_q > 255 (compiler_issues/grid_hard_limits.md).
#
# The GCU rewrite is a row-streaming 1-D kernel — the pattern proven by the
# enflame vit_attention / kpool / ple_fusion adaptations — with one twist
# for latency: each CTA owns HB heads of a single (b, m, kv-group), so the
# K gathered kv rows are loaded once per CTA and the HB online-softmax
# states run as independent interleaved chains (explicit unroll; Triton
# cannot carry list state, and 2-way kv unrolls measured slower). The
# general kernel keeps HB=4; the f8/f16 specializations below widen the
# batch when the head group divides evenly (see the dispatch note). Scores
# accumulate in fp32 with plain 1-D vector math, the softmax is scalar, and
# row bases are widened to int64. No tl.dot, no 2-D tiles, no
# reshape/broadcast: DSM stays ~30 KB per CTA and grid.y never carries SQ
# (flat chunk grid folded by _row_grid within grid.x<=65535 / y,z<=255).
#
# Autotune exemption (workflow.md G3): the perf-relevant knobs are the HB
# head batching and num_warps; @triton.autotune would multiply cold compiles
# under the current toolchain without a config space worth exploring.
# num_warps=1 measured fastest (nw=4 loses ~1.6x to the per-reduction
# cross-warp barrier; nw=2 loses ~1.5x and changes the reduction split).
# HB follows a host-side divisibility dispatch instead of autotune: the
# full-tile specializations f16/f8 (G % 16 == 0 / G % 8 == 0, D and TD
# exactly tiled) batch 16/8 heads per CTA so each gathered kv row feeds
# 4x more independent online-softmax chains — measured 1.31x/1.22x over
# the masked HB=4 form on the core shapes (2026-11-11, job
# j20261111190602_7fa6fae8); HB=32 regressed the small shapes and
# num_warps>=2 lost everywhere, both vetoed.

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)

_GRID_X_MAX = 65535
_GRID_Y_MAX = 255
_INT32_SAFE = 2**31
_HB = 4


def _row_grid(m):
    """Spread ``m`` logical chunks over a 3-D grid within GCU launch limits.

    The kernel recombines (pid_x, pid_y, pid_z) into a flat chunk id via the
    GRID_X / GRID_XY runtime arguments, so any decomposition with
    ``grid_x * grid_y * grid_z >= m`` is correct; chunks beyond ``m`` are
    masked out by the in-kernel ``pid < m`` guard. Same helper as the
    enflame fused_add_rms_norm adaptation.
    """
    if m <= _GRID_X_MAX:
        return (m, 1, 1)

    grid_y = triton.cdiv(m, _GRID_X_MAX)
    if grid_y <= _GRID_Y_MAX:
        return (triton.cdiv(m, grid_y), grid_y, 1)

    grid_z = triton.cdiv(m, _GRID_X_MAX * _GRID_Y_MAX)
    grid_y = triton.cdiv(m, _GRID_X_MAX * grid_z)
    return (triton.cdiv(m, grid_y * grid_z), grid_y, grid_z)


@triton.jit
def _mla_online_update(m_run, sum_e, acc, q_d, q_t, k_d, k_t, sm_scale):
    """One kv row's contribution to one head's online softmax state."""
    LOG2E: tl.constexpr = 1.4426950408889634
    score = tl.sum(q_d * k_d, axis=0) + tl.sum(q_t * k_t, axis=0)
    score = score * sm_scale
    m_new = tl.maximum(m_run, score)
    alpha = tl.exp2((m_run - m_new) * LOG2E)
    p = tl.exp2((score - m_new) * LOG2E)
    sum_e = sum_e * alpha + p
    acc = acc * alpha + p * k_d
    return m_new, sum_e, acc


@libentry()
@triton.jit
def sparse_mla_fwd_row_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    o_ptr,
    lse_ptr,
    sm_scale,
    TOTAL,
    SQ,
    H,
    VG,
    G,
    SKV,
    K,
    HC,
    GRID_X,
    GRID_XY,
    D: tl.constexpr,
    TD: tl.constexpr,
    DT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_TD: tl.constexpr,
    HB: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634
    pid = tl.program_id(0) + tl.program_id(1) * GRID_X + tl.program_id(2) * GRID_XY
    if pid < TOTAL:
        # chunk decode: pid -> (b, m, kv-group g, head-chunk hc); the HB
        # heads [g*G + hc*HB, +HB) share the indices row and kv group.
        hc = pid % HC
        tmg = pid // HC
        g = tmg % VG
        tm = tmg // VG
        b = tm // SQ
        m = tm - b * SQ
        h0 = g * G + hc * HB

        offs_d = tl.arange(0, BLOCK_D)
        offs_t = tl.arange(0, BLOCK_TD)
        mask_d = offs_d < D
        mask_t = offs_t < TD
        r0 = hc * HB + 0 < G
        r1 = hc * HB + 1 < G
        r2 = hc * HB + 2 < G
        r3 = hc * HB + 3 < G

        q_base = q_ptr + (tm * H + h0).to(tl.int64) * DT
        q_d0 = tl.load(q_base + offs_d, mask=mask_d & r0, other=0.0).to(tl.float32)
        q_t0 = tl.load(q_base + D + offs_t, mask=mask_t & r0, other=0.0).to(tl.float32)
        q_d1 = tl.load(q_base + DT + offs_d, mask=mask_d & r1, other=0.0).to(tl.float32)
        q_t1 = tl.load(q_base + DT + D + offs_t, mask=mask_t & r1, other=0.0).to(
            tl.float32
        )
        q_d2 = tl.load(q_base + 2 * DT + offs_d, mask=mask_d & r2, other=0.0).to(
            tl.float32
        )
        q_t2 = tl.load(q_base + 2 * DT + D + offs_t, mask=mask_t & r2, other=0.0).to(
            tl.float32
        )
        q_d3 = tl.load(q_base + 3 * DT + offs_d, mask=mask_d & r3, other=0.0).to(
            tl.float32
        )
        q_t3 = tl.load(q_base + 3 * DT + D + offs_t, mask=mask_t & r3, other=0.0).to(
            tl.float32
        )

        m0 = float("-inf")
        s0 = 0.0
        a0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m1 = float("-inf")
        s1 = 0.0
        a1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m2 = float("-inf")
        s2 = 0.0
        a2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m3 = float("-inf")
        s3 = 0.0
        a3 = tl.zeros((BLOCK_D,), dtype=tl.float32)

        t_base = idx_ptr + (tm * VG + g).to(tl.int64) * K
        for j in range(K):
            n = tl.load(t_base + j)
            if (n >= 0) & (n <= m):
                kv_row = kv_ptr + ((b * SKV + n) * VG + g).to(tl.int64) * DT
                k_d = tl.load(kv_row + offs_d, mask=mask_d, other=0.0).to(tl.float32)
                k_t = tl.load(kv_row + D + offs_t, mask=mask_t, other=0.0).to(
                    tl.float32
                )
                m0, s0, a0 = _mla_online_update(
                    m0, s0, a0, q_d0, q_t0, k_d, k_t, sm_scale
                )
                m1, s1, a1 = _mla_online_update(
                    m1, s1, a1, q_d1, q_t1, k_d, k_t, sm_scale
                )
                m2, s2, a2 = _mla_online_update(
                    m2, s2, a2, q_d2, q_t2, k_d, k_t, sm_scale
                )
                m3, s3, a3 = _mla_online_update(
                    m3, s3, a3, q_d3, q_t3, k_d, k_t, sm_scale
                )

        o_base = o_ptr + (tm * H + h0).to(tl.int64) * D
        l_base = lse_ptr + (tm * H + h0).to(tl.int64)
        if r0:
            inv = tl.where(s0 > 0.0, 1.0 / s0, 0.0)
            tl.store(
                o_base + offs_d, (a0 * inv).to(o_ptr.dtype.element_ty), mask=mask_d
            )
            tl.store(l_base, (m0 * LOG2E + tl.log2(s0)).to(lse_ptr.dtype.element_ty))
        if r1:
            inv = tl.where(s1 > 0.0, 1.0 / s1, 0.0)
            tl.store(
                o_base + D + offs_d,
                (a1 * inv).to(o_ptr.dtype.element_ty),
                mask=mask_d,
            )
            tl.store(
                l_base + 1, (m1 * LOG2E + tl.log2(s1)).to(lse_ptr.dtype.element_ty)
            )
        if r2:
            inv = tl.where(s2 > 0.0, 1.0 / s2, 0.0)
            tl.store(
                o_base + 2 * D + offs_d,
                (a2 * inv).to(o_ptr.dtype.element_ty),
                mask=mask_d,
            )
            tl.store(
                l_base + 2, (m2 * LOG2E + tl.log2(s2)).to(lse_ptr.dtype.element_ty)
            )
        if r3:
            inv = tl.where(s3 > 0.0, 1.0 / s3, 0.0)
            tl.store(
                o_base + 3 * D + offs_d,
                (a3 * inv).to(o_ptr.dtype.element_ty),
                mask=mask_d,
            )
            tl.store(
                l_base + 3, (m3 * LOG2E + tl.log2(s3)).to(lse_ptr.dtype.element_ty)
            )


@libentry()
@triton.jit
def sparse_mla_fwd_row_kernel_f8(
    q_ptr,
    kv_ptr,
    idx_ptr,
    o_ptr,
    lse_ptr,
    sm_scale,
    TOTAL,
    SQ,
    H,
    VG,
    G,
    SKV,
    K,
    HC,
    GRID_X,
    GRID_XY,
    D: tl.constexpr,
    TD: tl.constexpr,
    DT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_TD: tl.constexpr,
    HB: tl.constexpr,
):
    """Full-tile specialization: G % HB == 0 and D == BLOCK_D, TD == BLOCK_TD.

    Same math as sparse_mla_fwd_row_kernel with the head batch unrolled 8
    ways; every load/store is unconditional (no r* head guards, no mask_d /
    mask_t predicates), which is what the GCU compiler schedules best.
    """
    LOG2E: tl.constexpr = 1.4426950408889634
    pid = tl.program_id(0) + tl.program_id(1) * GRID_X + tl.program_id(2) * GRID_XY
    if pid < TOTAL:
        hc = pid % HC
        tmg = pid // HC
        g = tmg % VG
        tm = tmg // VG
        b = tm // SQ
        m = tm - b * SQ
        h0 = g * G + hc * HB

        offs_d = tl.arange(0, BLOCK_D)
        offs_t = tl.arange(0, BLOCK_TD)

        q_base = q_ptr + (tm * H + h0).to(tl.int64) * DT
        q_d0 = tl.load(q_base + 0 * DT + offs_d).to(tl.float32)
        q_t0 = tl.load(q_base + 0 * DT + D + offs_t).to(tl.float32)
        q_d1 = tl.load(q_base + 1 * DT + offs_d).to(tl.float32)
        q_t1 = tl.load(q_base + 1 * DT + D + offs_t).to(tl.float32)
        q_d2 = tl.load(q_base + 2 * DT + offs_d).to(tl.float32)
        q_t2 = tl.load(q_base + 2 * DT + D + offs_t).to(tl.float32)
        q_d3 = tl.load(q_base + 3 * DT + offs_d).to(tl.float32)
        q_t3 = tl.load(q_base + 3 * DT + D + offs_t).to(tl.float32)
        q_d4 = tl.load(q_base + 4 * DT + offs_d).to(tl.float32)
        q_t4 = tl.load(q_base + 4 * DT + D + offs_t).to(tl.float32)
        q_d5 = tl.load(q_base + 5 * DT + offs_d).to(tl.float32)
        q_t5 = tl.load(q_base + 5 * DT + D + offs_t).to(tl.float32)
        q_d6 = tl.load(q_base + 6 * DT + offs_d).to(tl.float32)
        q_t6 = tl.load(q_base + 6 * DT + D + offs_t).to(tl.float32)
        q_d7 = tl.load(q_base + 7 * DT + offs_d).to(tl.float32)
        q_t7 = tl.load(q_base + 7 * DT + D + offs_t).to(tl.float32)

        m0 = float("-inf")
        s0 = 0.0
        a0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m1 = float("-inf")
        s1 = 0.0
        a1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m2 = float("-inf")
        s2 = 0.0
        a2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m3 = float("-inf")
        s3 = 0.0
        a3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m4 = float("-inf")
        s4 = 0.0
        a4 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m5 = float("-inf")
        s5 = 0.0
        a5 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m6 = float("-inf")
        s6 = 0.0
        a6 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m7 = float("-inf")
        s7 = 0.0
        a7 = tl.zeros((BLOCK_D,), dtype=tl.float32)

        t_base = idx_ptr + (tm * VG + g).to(tl.int64) * K
        for j in range(K):
            n = tl.load(t_base + j)
            if (n >= 0) & (n <= m):
                kv_row = kv_ptr + ((b * SKV + n) * VG + g).to(tl.int64) * DT
                k_d = tl.load(kv_row + offs_d).to(tl.float32)
                k_t = tl.load(kv_row + D + offs_t).to(tl.float32)
                sc0 = (
                    tl.sum(q_d0 * k_d, axis=0) + tl.sum(q_t0 * k_t, axis=0)
                ) * sm_scale
                mn0 = tl.maximum(m0, sc0)
                al0 = tl.exp2((m0 - mn0) * LOG2E)
                p0 = tl.exp2((sc0 - mn0) * LOG2E)
                s0 = s0 * al0 + p0
                a0 = a0 * al0 + p0 * k_d
                m0 = mn0
                sc1 = (
                    tl.sum(q_d1 * k_d, axis=0) + tl.sum(q_t1 * k_t, axis=0)
                ) * sm_scale
                mn1 = tl.maximum(m1, sc1)
                al1 = tl.exp2((m1 - mn1) * LOG2E)
                p1 = tl.exp2((sc1 - mn1) * LOG2E)
                s1 = s1 * al1 + p1
                a1 = a1 * al1 + p1 * k_d
                m1 = mn1
                sc2 = (
                    tl.sum(q_d2 * k_d, axis=0) + tl.sum(q_t2 * k_t, axis=0)
                ) * sm_scale
                mn2 = tl.maximum(m2, sc2)
                al2 = tl.exp2((m2 - mn2) * LOG2E)
                p2 = tl.exp2((sc2 - mn2) * LOG2E)
                s2 = s2 * al2 + p2
                a2 = a2 * al2 + p2 * k_d
                m2 = mn2
                sc3 = (
                    tl.sum(q_d3 * k_d, axis=0) + tl.sum(q_t3 * k_t, axis=0)
                ) * sm_scale
                mn3 = tl.maximum(m3, sc3)
                al3 = tl.exp2((m3 - mn3) * LOG2E)
                p3 = tl.exp2((sc3 - mn3) * LOG2E)
                s3 = s3 * al3 + p3
                a3 = a3 * al3 + p3 * k_d
                m3 = mn3
                sc4 = (
                    tl.sum(q_d4 * k_d, axis=0) + tl.sum(q_t4 * k_t, axis=0)
                ) * sm_scale
                mn4 = tl.maximum(m4, sc4)
                al4 = tl.exp2((m4 - mn4) * LOG2E)
                p4 = tl.exp2((sc4 - mn4) * LOG2E)
                s4 = s4 * al4 + p4
                a4 = a4 * al4 + p4 * k_d
                m4 = mn4
                sc5 = (
                    tl.sum(q_d5 * k_d, axis=0) + tl.sum(q_t5 * k_t, axis=0)
                ) * sm_scale
                mn5 = tl.maximum(m5, sc5)
                al5 = tl.exp2((m5 - mn5) * LOG2E)
                p5 = tl.exp2((sc5 - mn5) * LOG2E)
                s5 = s5 * al5 + p5
                a5 = a5 * al5 + p5 * k_d
                m5 = mn5
                sc6 = (
                    tl.sum(q_d6 * k_d, axis=0) + tl.sum(q_t6 * k_t, axis=0)
                ) * sm_scale
                mn6 = tl.maximum(m6, sc6)
                al6 = tl.exp2((m6 - mn6) * LOG2E)
                p6 = tl.exp2((sc6 - mn6) * LOG2E)
                s6 = s6 * al6 + p6
                a6 = a6 * al6 + p6 * k_d
                m6 = mn6
                sc7 = (
                    tl.sum(q_d7 * k_d, axis=0) + tl.sum(q_t7 * k_t, axis=0)
                ) * sm_scale
                mn7 = tl.maximum(m7, sc7)
                al7 = tl.exp2((m7 - mn7) * LOG2E)
                p7 = tl.exp2((sc7 - mn7) * LOG2E)
                s7 = s7 * al7 + p7
                a7 = a7 * al7 + p7 * k_d
                m7 = mn7

        o_base = o_ptr + (tm * H + h0).to(tl.int64) * D
        l_base = lse_ptr + (tm * H + h0).to(tl.int64)
        inv0 = tl.where(s0 > 0.0, 1.0 / s0, 0.0)
        tl.store(o_base + 0 * D + offs_d, (a0 * inv0).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 0, (m0 * LOG2E + tl.log2(s0)).to(lse_ptr.dtype.element_ty))
        inv1 = tl.where(s1 > 0.0, 1.0 / s1, 0.0)
        tl.store(o_base + 1 * D + offs_d, (a1 * inv1).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 1, (m1 * LOG2E + tl.log2(s1)).to(lse_ptr.dtype.element_ty))
        inv2 = tl.where(s2 > 0.0, 1.0 / s2, 0.0)
        tl.store(o_base + 2 * D + offs_d, (a2 * inv2).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 2, (m2 * LOG2E + tl.log2(s2)).to(lse_ptr.dtype.element_ty))
        inv3 = tl.where(s3 > 0.0, 1.0 / s3, 0.0)
        tl.store(o_base + 3 * D + offs_d, (a3 * inv3).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 3, (m3 * LOG2E + tl.log2(s3)).to(lse_ptr.dtype.element_ty))
        inv4 = tl.where(s4 > 0.0, 1.0 / s4, 0.0)
        tl.store(o_base + 4 * D + offs_d, (a4 * inv4).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 4, (m4 * LOG2E + tl.log2(s4)).to(lse_ptr.dtype.element_ty))
        inv5 = tl.where(s5 > 0.0, 1.0 / s5, 0.0)
        tl.store(o_base + 5 * D + offs_d, (a5 * inv5).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 5, (m5 * LOG2E + tl.log2(s5)).to(lse_ptr.dtype.element_ty))
        inv6 = tl.where(s6 > 0.0, 1.0 / s6, 0.0)
        tl.store(o_base + 6 * D + offs_d, (a6 * inv6).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 6, (m6 * LOG2E + tl.log2(s6)).to(lse_ptr.dtype.element_ty))
        inv7 = tl.where(s7 > 0.0, 1.0 / s7, 0.0)
        tl.store(o_base + 7 * D + offs_d, (a7 * inv7).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 7, (m7 * LOG2E + tl.log2(s7)).to(lse_ptr.dtype.element_ty))


@libentry()
@triton.jit
def sparse_mla_fwd_row_kernel_f16(
    q_ptr,
    kv_ptr,
    idx_ptr,
    o_ptr,
    lse_ptr,
    sm_scale,
    TOTAL,
    SQ,
    H,
    VG,
    G,
    SKV,
    K,
    HC,
    GRID_X,
    GRID_XY,
    D: tl.constexpr,
    TD: tl.constexpr,
    DT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_TD: tl.constexpr,
    HB: tl.constexpr,
):
    """Full-tile specialization: G % HB == 0 and D == BLOCK_D, TD == BLOCK_TD.

    Same math as sparse_mla_fwd_row_kernel with the head batch unrolled 16
    ways; every load/store is unconditional (no r* head guards, no mask_d /
    mask_t predicates), which is what the GCU compiler schedules best.
    """
    LOG2E: tl.constexpr = 1.4426950408889634
    pid = tl.program_id(0) + tl.program_id(1) * GRID_X + tl.program_id(2) * GRID_XY
    if pid < TOTAL:
        hc = pid % HC
        tmg = pid // HC
        g = tmg % VG
        tm = tmg // VG
        b = tm // SQ
        m = tm - b * SQ
        h0 = g * G + hc * HB

        offs_d = tl.arange(0, BLOCK_D)
        offs_t = tl.arange(0, BLOCK_TD)

        q_base = q_ptr + (tm * H + h0).to(tl.int64) * DT
        q_d0 = tl.load(q_base + 0 * DT + offs_d).to(tl.float32)
        q_t0 = tl.load(q_base + 0 * DT + D + offs_t).to(tl.float32)
        q_d1 = tl.load(q_base + 1 * DT + offs_d).to(tl.float32)
        q_t1 = tl.load(q_base + 1 * DT + D + offs_t).to(tl.float32)
        q_d2 = tl.load(q_base + 2 * DT + offs_d).to(tl.float32)
        q_t2 = tl.load(q_base + 2 * DT + D + offs_t).to(tl.float32)
        q_d3 = tl.load(q_base + 3 * DT + offs_d).to(tl.float32)
        q_t3 = tl.load(q_base + 3 * DT + D + offs_t).to(tl.float32)
        q_d4 = tl.load(q_base + 4 * DT + offs_d).to(tl.float32)
        q_t4 = tl.load(q_base + 4 * DT + D + offs_t).to(tl.float32)
        q_d5 = tl.load(q_base + 5 * DT + offs_d).to(tl.float32)
        q_t5 = tl.load(q_base + 5 * DT + D + offs_t).to(tl.float32)
        q_d6 = tl.load(q_base + 6 * DT + offs_d).to(tl.float32)
        q_t6 = tl.load(q_base + 6 * DT + D + offs_t).to(tl.float32)
        q_d7 = tl.load(q_base + 7 * DT + offs_d).to(tl.float32)
        q_t7 = tl.load(q_base + 7 * DT + D + offs_t).to(tl.float32)
        q_d8 = tl.load(q_base + 8 * DT + offs_d).to(tl.float32)
        q_t8 = tl.load(q_base + 8 * DT + D + offs_t).to(tl.float32)
        q_d9 = tl.load(q_base + 9 * DT + offs_d).to(tl.float32)
        q_t9 = tl.load(q_base + 9 * DT + D + offs_t).to(tl.float32)
        q_d10 = tl.load(q_base + 10 * DT + offs_d).to(tl.float32)
        q_t10 = tl.load(q_base + 10 * DT + D + offs_t).to(tl.float32)
        q_d11 = tl.load(q_base + 11 * DT + offs_d).to(tl.float32)
        q_t11 = tl.load(q_base + 11 * DT + D + offs_t).to(tl.float32)
        q_d12 = tl.load(q_base + 12 * DT + offs_d).to(tl.float32)
        q_t12 = tl.load(q_base + 12 * DT + D + offs_t).to(tl.float32)
        q_d13 = tl.load(q_base + 13 * DT + offs_d).to(tl.float32)
        q_t13 = tl.load(q_base + 13 * DT + D + offs_t).to(tl.float32)
        q_d14 = tl.load(q_base + 14 * DT + offs_d).to(tl.float32)
        q_t14 = tl.load(q_base + 14 * DT + D + offs_t).to(tl.float32)
        q_d15 = tl.load(q_base + 15 * DT + offs_d).to(tl.float32)
        q_t15 = tl.load(q_base + 15 * DT + D + offs_t).to(tl.float32)

        m0 = float("-inf")
        s0 = 0.0
        a0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m1 = float("-inf")
        s1 = 0.0
        a1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m2 = float("-inf")
        s2 = 0.0
        a2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m3 = float("-inf")
        s3 = 0.0
        a3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m4 = float("-inf")
        s4 = 0.0
        a4 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m5 = float("-inf")
        s5 = 0.0
        a5 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m6 = float("-inf")
        s6 = 0.0
        a6 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m7 = float("-inf")
        s7 = 0.0
        a7 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m8 = float("-inf")
        s8 = 0.0
        a8 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m9 = float("-inf")
        s9 = 0.0
        a9 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m10 = float("-inf")
        s10 = 0.0
        a10 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m11 = float("-inf")
        s11 = 0.0
        a11 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m12 = float("-inf")
        s12 = 0.0
        a12 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m13 = float("-inf")
        s13 = 0.0
        a13 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m14 = float("-inf")
        s14 = 0.0
        a14 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m15 = float("-inf")
        s15 = 0.0
        a15 = tl.zeros((BLOCK_D,), dtype=tl.float32)

        t_base = idx_ptr + (tm * VG + g).to(tl.int64) * K
        for j in range(K):
            n = tl.load(t_base + j)
            if (n >= 0) & (n <= m):
                kv_row = kv_ptr + ((b * SKV + n) * VG + g).to(tl.int64) * DT
                k_d = tl.load(kv_row + offs_d).to(tl.float32)
                k_t = tl.load(kv_row + D + offs_t).to(tl.float32)
                sc0 = (
                    tl.sum(q_d0 * k_d, axis=0) + tl.sum(q_t0 * k_t, axis=0)
                ) * sm_scale
                mn0 = tl.maximum(m0, sc0)
                al0 = tl.exp2((m0 - mn0) * LOG2E)
                p0 = tl.exp2((sc0 - mn0) * LOG2E)
                s0 = s0 * al0 + p0
                a0 = a0 * al0 + p0 * k_d
                m0 = mn0
                sc1 = (
                    tl.sum(q_d1 * k_d, axis=0) + tl.sum(q_t1 * k_t, axis=0)
                ) * sm_scale
                mn1 = tl.maximum(m1, sc1)
                al1 = tl.exp2((m1 - mn1) * LOG2E)
                p1 = tl.exp2((sc1 - mn1) * LOG2E)
                s1 = s1 * al1 + p1
                a1 = a1 * al1 + p1 * k_d
                m1 = mn1
                sc2 = (
                    tl.sum(q_d2 * k_d, axis=0) + tl.sum(q_t2 * k_t, axis=0)
                ) * sm_scale
                mn2 = tl.maximum(m2, sc2)
                al2 = tl.exp2((m2 - mn2) * LOG2E)
                p2 = tl.exp2((sc2 - mn2) * LOG2E)
                s2 = s2 * al2 + p2
                a2 = a2 * al2 + p2 * k_d
                m2 = mn2
                sc3 = (
                    tl.sum(q_d3 * k_d, axis=0) + tl.sum(q_t3 * k_t, axis=0)
                ) * sm_scale
                mn3 = tl.maximum(m3, sc3)
                al3 = tl.exp2((m3 - mn3) * LOG2E)
                p3 = tl.exp2((sc3 - mn3) * LOG2E)
                s3 = s3 * al3 + p3
                a3 = a3 * al3 + p3 * k_d
                m3 = mn3
                sc4 = (
                    tl.sum(q_d4 * k_d, axis=0) + tl.sum(q_t4 * k_t, axis=0)
                ) * sm_scale
                mn4 = tl.maximum(m4, sc4)
                al4 = tl.exp2((m4 - mn4) * LOG2E)
                p4 = tl.exp2((sc4 - mn4) * LOG2E)
                s4 = s4 * al4 + p4
                a4 = a4 * al4 + p4 * k_d
                m4 = mn4
                sc5 = (
                    tl.sum(q_d5 * k_d, axis=0) + tl.sum(q_t5 * k_t, axis=0)
                ) * sm_scale
                mn5 = tl.maximum(m5, sc5)
                al5 = tl.exp2((m5 - mn5) * LOG2E)
                p5 = tl.exp2((sc5 - mn5) * LOG2E)
                s5 = s5 * al5 + p5
                a5 = a5 * al5 + p5 * k_d
                m5 = mn5
                sc6 = (
                    tl.sum(q_d6 * k_d, axis=0) + tl.sum(q_t6 * k_t, axis=0)
                ) * sm_scale
                mn6 = tl.maximum(m6, sc6)
                al6 = tl.exp2((m6 - mn6) * LOG2E)
                p6 = tl.exp2((sc6 - mn6) * LOG2E)
                s6 = s6 * al6 + p6
                a6 = a6 * al6 + p6 * k_d
                m6 = mn6
                sc7 = (
                    tl.sum(q_d7 * k_d, axis=0) + tl.sum(q_t7 * k_t, axis=0)
                ) * sm_scale
                mn7 = tl.maximum(m7, sc7)
                al7 = tl.exp2((m7 - mn7) * LOG2E)
                p7 = tl.exp2((sc7 - mn7) * LOG2E)
                s7 = s7 * al7 + p7
                a7 = a7 * al7 + p7 * k_d
                m7 = mn7
                sc8 = (
                    tl.sum(q_d8 * k_d, axis=0) + tl.sum(q_t8 * k_t, axis=0)
                ) * sm_scale
                mn8 = tl.maximum(m8, sc8)
                al8 = tl.exp2((m8 - mn8) * LOG2E)
                p8 = tl.exp2((sc8 - mn8) * LOG2E)
                s8 = s8 * al8 + p8
                a8 = a8 * al8 + p8 * k_d
                m8 = mn8
                sc9 = (
                    tl.sum(q_d9 * k_d, axis=0) + tl.sum(q_t9 * k_t, axis=0)
                ) * sm_scale
                mn9 = tl.maximum(m9, sc9)
                al9 = tl.exp2((m9 - mn9) * LOG2E)
                p9 = tl.exp2((sc9 - mn9) * LOG2E)
                s9 = s9 * al9 + p9
                a9 = a9 * al9 + p9 * k_d
                m9 = mn9
                sc10 = (
                    tl.sum(q_d10 * k_d, axis=0) + tl.sum(q_t10 * k_t, axis=0)
                ) * sm_scale
                mn10 = tl.maximum(m10, sc10)
                al10 = tl.exp2((m10 - mn10) * LOG2E)
                p10 = tl.exp2((sc10 - mn10) * LOG2E)
                s10 = s10 * al10 + p10
                a10 = a10 * al10 + p10 * k_d
                m10 = mn10
                sc11 = (
                    tl.sum(q_d11 * k_d, axis=0) + tl.sum(q_t11 * k_t, axis=0)
                ) * sm_scale
                mn11 = tl.maximum(m11, sc11)
                al11 = tl.exp2((m11 - mn11) * LOG2E)
                p11 = tl.exp2((sc11 - mn11) * LOG2E)
                s11 = s11 * al11 + p11
                a11 = a11 * al11 + p11 * k_d
                m11 = mn11
                sc12 = (
                    tl.sum(q_d12 * k_d, axis=0) + tl.sum(q_t12 * k_t, axis=0)
                ) * sm_scale
                mn12 = tl.maximum(m12, sc12)
                al12 = tl.exp2((m12 - mn12) * LOG2E)
                p12 = tl.exp2((sc12 - mn12) * LOG2E)
                s12 = s12 * al12 + p12
                a12 = a12 * al12 + p12 * k_d
                m12 = mn12
                sc13 = (
                    tl.sum(q_d13 * k_d, axis=0) + tl.sum(q_t13 * k_t, axis=0)
                ) * sm_scale
                mn13 = tl.maximum(m13, sc13)
                al13 = tl.exp2((m13 - mn13) * LOG2E)
                p13 = tl.exp2((sc13 - mn13) * LOG2E)
                s13 = s13 * al13 + p13
                a13 = a13 * al13 + p13 * k_d
                m13 = mn13
                sc14 = (
                    tl.sum(q_d14 * k_d, axis=0) + tl.sum(q_t14 * k_t, axis=0)
                ) * sm_scale
                mn14 = tl.maximum(m14, sc14)
                al14 = tl.exp2((m14 - mn14) * LOG2E)
                p14 = tl.exp2((sc14 - mn14) * LOG2E)
                s14 = s14 * al14 + p14
                a14 = a14 * al14 + p14 * k_d
                m14 = mn14
                sc15 = (
                    tl.sum(q_d15 * k_d, axis=0) + tl.sum(q_t15 * k_t, axis=0)
                ) * sm_scale
                mn15 = tl.maximum(m15, sc15)
                al15 = tl.exp2((m15 - mn15) * LOG2E)
                p15 = tl.exp2((sc15 - mn15) * LOG2E)
                s15 = s15 * al15 + p15
                a15 = a15 * al15 + p15 * k_d
                m15 = mn15

        o_base = o_ptr + (tm * H + h0).to(tl.int64) * D
        l_base = lse_ptr + (tm * H + h0).to(tl.int64)
        inv0 = tl.where(s0 > 0.0, 1.0 / s0, 0.0)
        tl.store(o_base + 0 * D + offs_d, (a0 * inv0).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 0, (m0 * LOG2E + tl.log2(s0)).to(lse_ptr.dtype.element_ty))
        inv1 = tl.where(s1 > 0.0, 1.0 / s1, 0.0)
        tl.store(o_base + 1 * D + offs_d, (a1 * inv1).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 1, (m1 * LOG2E + tl.log2(s1)).to(lse_ptr.dtype.element_ty))
        inv2 = tl.where(s2 > 0.0, 1.0 / s2, 0.0)
        tl.store(o_base + 2 * D + offs_d, (a2 * inv2).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 2, (m2 * LOG2E + tl.log2(s2)).to(lse_ptr.dtype.element_ty))
        inv3 = tl.where(s3 > 0.0, 1.0 / s3, 0.0)
        tl.store(o_base + 3 * D + offs_d, (a3 * inv3).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 3, (m3 * LOG2E + tl.log2(s3)).to(lse_ptr.dtype.element_ty))
        inv4 = tl.where(s4 > 0.0, 1.0 / s4, 0.0)
        tl.store(o_base + 4 * D + offs_d, (a4 * inv4).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 4, (m4 * LOG2E + tl.log2(s4)).to(lse_ptr.dtype.element_ty))
        inv5 = tl.where(s5 > 0.0, 1.0 / s5, 0.0)
        tl.store(o_base + 5 * D + offs_d, (a5 * inv5).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 5, (m5 * LOG2E + tl.log2(s5)).to(lse_ptr.dtype.element_ty))
        inv6 = tl.where(s6 > 0.0, 1.0 / s6, 0.0)
        tl.store(o_base + 6 * D + offs_d, (a6 * inv6).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 6, (m6 * LOG2E + tl.log2(s6)).to(lse_ptr.dtype.element_ty))
        inv7 = tl.where(s7 > 0.0, 1.0 / s7, 0.0)
        tl.store(o_base + 7 * D + offs_d, (a7 * inv7).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 7, (m7 * LOG2E + tl.log2(s7)).to(lse_ptr.dtype.element_ty))
        inv8 = tl.where(s8 > 0.0, 1.0 / s8, 0.0)
        tl.store(o_base + 8 * D + offs_d, (a8 * inv8).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 8, (m8 * LOG2E + tl.log2(s8)).to(lse_ptr.dtype.element_ty))
        inv9 = tl.where(s9 > 0.0, 1.0 / s9, 0.0)
        tl.store(o_base + 9 * D + offs_d, (a9 * inv9).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 9, (m9 * LOG2E + tl.log2(s9)).to(lse_ptr.dtype.element_ty))
        inv10 = tl.where(s10 > 0.0, 1.0 / s10, 0.0)
        tl.store(o_base + 10 * D + offs_d, (a10 * inv10).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 10, (m10 * LOG2E + tl.log2(s10)).to(lse_ptr.dtype.element_ty))
        inv11 = tl.where(s11 > 0.0, 1.0 / s11, 0.0)
        tl.store(o_base + 11 * D + offs_d, (a11 * inv11).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 11, (m11 * LOG2E + tl.log2(s11)).to(lse_ptr.dtype.element_ty))
        inv12 = tl.where(s12 > 0.0, 1.0 / s12, 0.0)
        tl.store(o_base + 12 * D + offs_d, (a12 * inv12).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 12, (m12 * LOG2E + tl.log2(s12)).to(lse_ptr.dtype.element_ty))
        inv13 = tl.where(s13 > 0.0, 1.0 / s13, 0.0)
        tl.store(o_base + 13 * D + offs_d, (a13 * inv13).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 13, (m13 * LOG2E + tl.log2(s13)).to(lse_ptr.dtype.element_ty))
        inv14 = tl.where(s14 > 0.0, 1.0 / s14, 0.0)
        tl.store(o_base + 14 * D + offs_d, (a14 * inv14).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 14, (m14 * LOG2E + tl.log2(s14)).to(lse_ptr.dtype.element_ty))
        inv15 = tl.where(s15 > 0.0, 1.0 / s15, 0.0)
        tl.store(o_base + 15 * D + offs_d, (a15 * inv15).to(o_ptr.dtype.element_ty))
        tl.store(l_base + 15, (m15 * LOG2E + tl.log2(s15)).to(lse_ptr.dtype.element_ty))


def triton_sparse_mla_fwd_interface(
    q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512
):
    logger.debug("GEMS SPARSE_MLA_FWD_INTERFACE")
    assert return_p_sum is False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    B, SQ, H, DT = q.shape
    _, SKV, VG, _ = kv.shape

    D = d_v
    assert 0 < D <= DT, "d_v must lie within the qkv head dim"
    TD = DT - D
    assert kv.shape[-1] == DT
    K = indices.shape[-1]
    assert indices.shape == (B, SQ, VG, K)
    G = H // VG
    if sm_scale is None:
        sm_scale = DT**-0.5

    total = B * SQ * H
    if (
        total >= _INT32_SAFE
        or B * SQ * VG >= _INT32_SAFE
        or B * SKV * VG >= _INT32_SAFE
    ):
        raise NotImplementedError(
            "sparse_mla row counts exceed the int32 safe domain of the GCU "
            "row-streaming kernel"
        )

    output = torch.zeros((B, SQ, H, D), device=q.device, dtype=q.dtype)
    lse = torch.full((B, SQ, H), float("-inf"), device=q.device, dtype=q.dtype)

    if total and K:
        block_d = triton.next_power_of_2(D)
        block_td = triton.next_power_of_2(max(TD, 1))
        # Full-tile fast path: when the head group divides the batch exactly
        # and the D/TD tiles need no masking, the specialized kernels drop
        # every predicate and widen the head batch to 16/8 (measured 1.31x /
        # 1.22x over the HB=4 masked form on the core shapes; outputs are
        # bit-identical). Anything else keeps the general masked HB=4 kernel.
        kernel = sparse_mla_fwd_row_kernel
        hb = _HB
        if block_d == D and block_td == TD:
            if G % 16 == 0:
                kernel = sparse_mla_fwd_row_kernel_f16
                hb = 16
            elif G % 8 == 0:
                kernel = sparse_mla_fwd_row_kernel_f8
                hb = 8
        hc = triton.cdiv(G, hb)
        total_chunks = B * SQ * VG * hc
        grid = _row_grid(total_chunks)
        with torch_device_fn.device(q.device):
            kernel[grid](
                q,
                kv,
                indices,
                output,
                lse,
                sm_scale,
                total_chunks,
                SQ,
                H,
                VG,
                G,
                SKV,
                K,
                hc,
                grid[0],
                grid[0] * grid[1],
                D=D,
                TD=TD,
                DT=DT,
                BLOCK_D=block_d,
                BLOCK_TD=block_td,
                HB=hb,
                num_warps=1,
            )
    return output, lse


def _install():
    """Wire the GCU implementation into the direct-import entrypoint.

    tests/test_DSA/test_sparse_mla_ops.py and
    benchmark/test_sparse_mla_fwd_interface.py import
    ``from flaggems_vllm.ops.DSA.sparse_mla import triton_sparse_mla_fwd_interface``,
    which the SpecOpRegistrar top-level namespace swap cannot reach. Unlike
    kpool, the generic module is not imported by ``flaggems_vllm.ops.__init__``,
    so import it here (idempotent) and replace the pristine attribute,
    mirroring the kpool / kunlunxin override precedents: only a pristine
    generic function is swapped, so explicit user or test patches always win.
    """
    import importlib
    import sys

    try:
        mod = sys.modules.get("flaggems_vllm.ops.DSA.sparse_mla")
        if mod is None:
            mod = importlib.import_module("flaggems_vllm.ops.DSA.sparse_mla")
    except Exception:
        return
    current = getattr(mod, "triton_sparse_mla_fwd_interface", None)
    if current is not None and getattr(current, "__module__", None) == (
        "flaggems_vllm.ops.DSA.sparse_mla"
    ):
        setattr(mod, "triton_sparse_mla_fwd_interface", triton_sparse_mla_fwd_interface)


try:
    _install()
except Exception:
    # pytest assertion-rewrite 上下文中 @triton.jit 源定位可能失败
    # (collect 期动态触碰 generic 模块触发);跳过接线不影响 op 路由
    # (SpecOpRegistrar 已保证),直连 import 入口的接线由运行期重试兜底。
    pass
__all__ = ["triton_sparse_mla_fwd_interface"]
