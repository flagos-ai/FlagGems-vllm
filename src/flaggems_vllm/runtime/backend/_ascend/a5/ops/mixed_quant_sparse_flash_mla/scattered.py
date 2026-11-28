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

"""Single-pass attention with the gather and dequant inlined.

These kernels read the FP8 pages straight into the attention loop, so a page is
re-fetched once per head group instead of once per query.  That is the wrong
trade when several head groups share a query, but at T1=1 there is only one
query and the extra traffic is bounded, while the gather pipeline would pay
two or three kernel launches plus a round trip through HBM for the staging
buffer.  Hence the split-K form here owns the T1=1 dispatch, and
``scattered_attn_kernel`` stays as the single-kernel reference that the
pipelined paths are checked against.
"""

import triton
import triton.language as tl

__all__ = ["scattered_attn_kernel", "scattered_splitk_main_kernel"]


@triton.jit
def scattered_attn_kernel(
    # ── Inputs ──
    q_ptr,  # [T1, N1_C, D_C] bf16
    ori_data_ptr,  # fp8_e4m3 [nrow, D_C], row pitch ROW_STRIDE_DATA
    ori_scale_ptr,  # bf16 [nrow, NG], row pitch ROW_STRIDE_SCALE
    ori_idx_ptr,  # [T1, K_TOTAL] int32
    ori_bt_ptr,  # [T1, MAX_BLK] int32
    ori_topk_ptr,  # [T1] int32, or a dummy when HAS_TOPK is False
    sinks_ptr,  # [N1_C] fp32, or a dummy when HAS_SINKS is False
    # ── Outputs ──
    out_ptr,  # [T1, N1_C, D_C] bf16
    lse_ptr,  # [T1, N1_C] fp32, written only when RETURN_LSE
    # ── Scalars ──
    softmax_scale,
    # ── Constexprs ──
    T1: tl.constexpr,
    N1_C: tl.constexpr,
    K_TOTAL: tl.constexpr,
    MAX_BLK: tl.constexpr,
    BS: tl.constexpr,
    LOG2_BS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NUM_CORES: tl.constexpr,
    D_C: tl.constexpr,
    NG: tl.constexpr,
    GS: tl.constexpr,
    ROW_STRIDE_DATA: tl.constexpr,
    ROW_STRIDE_SCALE: tl.constexpr,
    HAS_TOPK: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Gather, dequantize and attend in one pass over the full K range.

    Grid is ``(NUM_CORES,)``; each core walks ``(t, head group)`` round-robin.
    """
    cid = tl.program_id(0)
    N_HEAD_GROUPS: tl.constexpr = N1_C // BLOCK_H
    TOTAL_TASKS = T1 * N_HEAD_GROUPS

    offs_d = tl.arange(0, D_C)
    offs_sc = tl.arange(0, NG)

    for task_idx in range(cid, TOTAL_TASKS, NUM_CORES):
        t = task_idx // N_HEAD_GROUPS
        hg = task_idx % N_HEAD_GROUPS
        offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)

        q = tl.load(q_ptr + t * N1_C * D_C + offs_h[:, None] * D_C + offs_d[None, :])

        if HAS_SINKS:
            m_i = tl.load(sinks_ptr + offs_h)
            l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
        else:
            m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
            l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, D_C], dtype=tl.float32)

        if HAS_TOPK:
            valid_cnt = tl.load(ori_topk_ptr + t).to(tl.int32)
        else:
            valid_cnt = K_TOTAL

        for s0 in range(0, K_TOTAL, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)

            idx = tl.load(ori_idx_ptr + t * K_TOTAL + offs_s)
            valid = (idx >= 0) & (offs_s < valid_cnt)
            idxc = tl.maximum(idx, 0)

            logical_blk = idxc >> LOG2_BS
            off_in_blk = idxc & (BS - 1)
            phys_blk = tl.load(ori_bt_ptr + t * MAX_BLK + logical_blk)
            row = phys_blk * BS + off_in_blk

            k_fp8 = tl.load(
                ori_data_ptr + row[:, None] * ROW_STRIDE_DATA + offs_d[None, :]
            )
            sc = tl.load(
                ori_scale_ptr + row[:, None] * ROW_STRIDE_SCALE + offs_sc[None, :]
            )

            k3 = tl.reshape(k_fp8.to(tl.bfloat16), (BLOCK_S, NG, GS))
            k_deq = tl.reshape(k3 * sc.to(tl.bfloat16)[:, :, None], (BLOCK_S, D_C))

            s = tl.dot(q, tl.trans(k_deq), out_dtype=tl.float32)
            s = s * softmax_scale
            s = tl.where(valid[None, :], s, float("-inf"))

            tile_max = tl.max(s, axis=1)
            tile_max = tl.maximum(tile_max, 0.0)
            m_new = tl.maximum(m_i, tile_max)
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
            p = tl.exp(s - m_safe[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16), k_deq, out_dtype=tl.float32
            )
            m_i = m_new

        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        o = acc / safe_l[:, None]
        tl.store(
            out_ptr + t * N1_C * D_C + offs_h[:, None] * D_C + offs_d[None, :],
            o.to(tl.bfloat16),
        )

        if RETURN_LSE:
            lse = m_i + tl.log(safe_l)
            tl.store(lse_ptr + t * N1_C + offs_h, lse)


@triton.jit
def scattered_splitk_main_kernel(
    # ── Inputs ──
    q_ptr,  # [T1, N1_C, D_C] bf16
    ori_data_ptr,  # fp8_e4m3 [nrow, D_C], row pitch ROW_STRIDE_DATA
    ori_scale_ptr,  # bf16 [nrow, NG], row pitch ROW_STRIDE_SCALE
    ori_idx_ptr,  # [T1, K_TOTAL] int32
    ori_bt_ptr,  # [T1, MAX_BLK] int32
    ori_topk_ptr,  # [T1] int32, or a dummy when HAS_TOPK is False
    sinks_ptr,  # [N1_C] fp32, or a dummy when HAS_SINKS is False
    # ── Workspace outputs ──
    ws_acc_ptr,  # [T1, N_HEAD_GROUPS, SPLITS, BLOCK_H, D_C] fp32
    ws_m_ptr,  # [T1, N_HEAD_GROUPS, SPLITS, BLOCK_H] fp32
    ws_l_ptr,  # [T1, N_HEAD_GROUPS, SPLITS, BLOCK_H] fp32
    # ── Scalars ──
    softmax_scale,
    # ── Constexprs ──
    T1: tl.constexpr,
    N1_C: tl.constexpr,
    K_TOTAL: tl.constexpr,
    MAX_BLK: tl.constexpr,
    BS: tl.constexpr,
    LOG2_BS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SPLITS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    D_C: tl.constexpr,
    NG: tl.constexpr,
    GS: tl.constexpr,
    ROW_STRIDE_DATA: tl.constexpr,
    ROW_STRIDE_SCALE: tl.constexpr,
    HAS_TOPK: tl.constexpr,
    HAS_SINKS: tl.constexpr,
):
    """Split-K form of ``scattered_attn_kernel``, writing per-split partials.

    Pairs with ``attention.splitk_combine_kernel``.  Each task owns
    ``(t, head group, split)`` and covers ``K_TOTAL // SPLITS`` tokens.
    """
    cid = tl.program_id(0)
    N_HEAD_GROUPS: tl.constexpr = N1_C // BLOCK_H
    TOTAL_TASKS = T1 * N_HEAD_GROUPS * SPLITS
    K_PER_SPLIT: tl.constexpr = K_TOTAL // SPLITS

    offs_d = tl.arange(0, D_C)
    offs_sc = tl.arange(0, NG)
    offs_bh = tl.arange(0, BLOCK_H)

    for task_idx in range(cid, TOTAL_TASKS, NUM_CORES):
        t = task_idx // (N_HEAD_GROUPS * SPLITS)
        remainder = task_idx % (N_HEAD_GROUPS * SPLITS)
        hg = remainder // SPLITS
        split_id = remainder % SPLITS
        offs_h = hg * BLOCK_H + offs_bh

        q = tl.load(q_ptr + t * N1_C * D_C + offs_h[:, None] * D_C + offs_d[None, :])

        if HAS_SINKS:
            if split_id == 0:
                m_i = tl.load(sinks_ptr + offs_h)
                l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
            else:
                m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
                l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        else:
            m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
            l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, D_C], dtype=tl.float32)

        if HAS_TOPK:
            valid_cnt = tl.load(ori_topk_ptr + t).to(tl.int32)
        else:
            valid_cnt = K_TOTAL

        k_start = split_id * K_PER_SPLIT

        for s0 in range(k_start, k_start + K_PER_SPLIT, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)

            idx = tl.load(ori_idx_ptr + t * K_TOTAL + offs_s)
            valid = (idx >= 0) & (offs_s < valid_cnt)
            idxc = tl.maximum(idx, 0)

            logical_blk = idxc >> LOG2_BS
            off_in_blk = idxc & (BS - 1)
            phys_blk = tl.load(ori_bt_ptr + t * MAX_BLK + logical_blk)
            row = phys_blk * BS + off_in_blk

            k_fp8 = tl.load(
                ori_data_ptr + row[:, None] * ROW_STRIDE_DATA + offs_d[None, :]
            )
            sc = tl.load(
                ori_scale_ptr + row[:, None] * ROW_STRIDE_SCALE + offs_sc[None, :]
            )

            k3 = tl.reshape(k_fp8.to(tl.bfloat16), (BLOCK_S, NG, GS))
            k_deq = tl.reshape(k3 * sc.to(tl.bfloat16)[:, :, None], (BLOCK_S, D_C))

            s = tl.dot(q, tl.trans(k_deq), out_dtype=tl.float32)
            s = s * softmax_scale
            s = tl.where(valid[None, :], s, float("-inf"))

            tile_max = tl.max(s, axis=1)
            tile_max = tl.maximum(tile_max, 0.0)
            m_new = tl.maximum(m_i, tile_max)
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
            p = tl.exp(s - m_safe[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16), k_deq, out_dtype=tl.float32
            )
            m_i = m_new

        ws_slot = (t * N_HEAD_GROUPS + hg) * SPLITS + split_id
        tl.store(
            ws_acc_ptr
            + ws_slot * BLOCK_H * D_C
            + offs_bh[:, None] * D_C
            + offs_d[None, :],
            acc,
        )
        tl.store(ws_m_ptr + ws_slot * BLOCK_H + offs_bh, m_i)
        tl.store(ws_l_ptr + ws_slot * BLOCK_H + offs_bh, l_i)
