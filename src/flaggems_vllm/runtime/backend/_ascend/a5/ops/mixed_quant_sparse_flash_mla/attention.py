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

"""Attention over the pre-gathered contiguous KV buffer.

Three kernels:

``contig_attn_kernel``
    One task per ``(token, head group)``, full K range.  Used when there are
    already at least as many tasks as cores.

``contig_splitk_main_kernel``
    One task per ``(token, head group, split)``, writing partial
    ``(acc, row_max, row_sum)`` to a workspace.  Used when the plain task count
    would leave cores idle.

``splitk_combine_kernel``
    Merges the per-split partials with the same online-softmax rescaling and
    writes the final output.  Shared with the scattered split-K path.

All of them take K == V (MLA absorb), so the dequantized row serves as both the
key for ``QK^T`` and the value for ``PV``; the tile is loaded once.  Sinks seed
``row_max`` with ``row_sum = 1``, which is ``exp(sink - sink)``, and only split
0 may apply them so the combine does not count them twice.
"""

import triton
import triton.language as tl

__all__ = [
    "contig_attn_kernel",
    "contig_splitk_main_kernel",
    "splitk_combine_kernel",
]


@triton.jit
def contig_attn_kernel(
    # ── Inputs ──
    q_ptr,  # [T1, N1_C, D_C] bf16
    deq_kv_ptr,  # [T1, K_TOTAL, D_C] bf16
    valid_ptr,  # [T1, K_TOTAL] int8
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
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NUM_CORES: tl.constexpr,
    D_C: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Attention from the contiguous dequantized buffer, no split over K.

    Grid is ``(NUM_CORES,)``; each core walks ``(t, head group)`` tasks
    round-robin.  Every access is sequential or strided, with no indirection.
    """
    cid = tl.program_id(0)
    N_HEAD_GROUPS: tl.constexpr = N1_C // BLOCK_H
    TOTAL_TASKS = T1 * N_HEAD_GROUPS

    offs_d = tl.arange(0, D_C)

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

        kv_base = t * K_TOTAL * D_C
        valid_base = t * K_TOTAL

        for s0 in range(0, K_TOTAL, BLOCK_S):
            offs_s = tl.arange(0, BLOCK_S)

            valid = tl.load(valid_ptr + valid_base + s0 + offs_s).to(tl.int1)
            kv_tile = tl.load(
                deq_kv_ptr + kv_base + (s0 + offs_s[:, None]) * D_C + offs_d[None, :]
            )

            s = tl.dot(q, tl.trans(kv_tile), out_dtype=tl.float32)
            s = s * softmax_scale
            s = tl.where(valid[None, :], s, float("-inf"))

            # Online softmax.  clamp_min(0) keeps the running max above the
            # implicit zero logit that the sink contributes.
            tile_max = tl.max(s, axis=1)
            tile_max = tl.maximum(tile_max, 0.0)
            m_new = tl.maximum(m_i, tile_max)
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
            p = tl.exp(s - m_safe[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16), kv_tile, out_dtype=tl.float32
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
def contig_splitk_main_kernel(
    # ── Inputs ──
    q_ptr,  # [T1, N1_C, D_C] bf16
    deq_kv_ptr,  # [T1, K_TOTAL, D_C] bf16
    valid_ptr,  # [T1, K_TOTAL] int8
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
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SPLITS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    D_C: tl.constexpr,
    HAS_SINKS: tl.constexpr,
):
    """Split-K attention from the contiguous buffer, writing partials.

    Each task owns ``(t, head group, split)`` and covers
    ``K_TOTAL // SPLITS`` tokens.  With T1=1, N1=64 and splits=8 this turns 4
    tasks into 32, which is the difference between 4 busy cores and all of them.
    """
    cid = tl.program_id(0)
    N_HEAD_GROUPS: tl.constexpr = N1_C // BLOCK_H
    TOTAL_TASKS = T1 * N_HEAD_GROUPS * SPLITS
    K_PER_SPLIT: tl.constexpr = K_TOTAL // SPLITS

    offs_d = tl.arange(0, D_C)
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

        kv_base = t * K_TOTAL * D_C
        valid_base = t * K_TOTAL
        k_start = split_id * K_PER_SPLIT

        for s0 in range(k_start, k_start + K_PER_SPLIT, BLOCK_S):
            offs_s = tl.arange(0, BLOCK_S)

            valid = tl.load(valid_ptr + valid_base + s0 + offs_s).to(tl.int1)
            kv_tile = tl.load(
                deq_kv_ptr + kv_base + (s0 + offs_s[:, None]) * D_C + offs_d[None, :]
            )

            s = tl.dot(q, tl.trans(kv_tile), out_dtype=tl.float32)
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
                p.to(tl.bfloat16), kv_tile, out_dtype=tl.float32
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


@triton.jit
def splitk_combine_kernel(
    # ── Workspace inputs ──
    ws_acc_ptr,  # [T1, N_HEAD_GROUPS, SPLITS, BLOCK_H, D_C] fp32
    ws_m_ptr,  # [T1, N_HEAD_GROUPS, SPLITS, BLOCK_H] fp32
    ws_l_ptr,  # [T1, N_HEAD_GROUPS, SPLITS, BLOCK_H] fp32
    # ── Outputs ──
    out_ptr,  # [T1, N1_C, D_C] bf16
    lse_ptr,  # [T1, N1_C] fp32, written only when RETURN_LSE
    # ── Constexprs ──
    T1: tl.constexpr,
    N1_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SPLITS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    D_C: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Merge the ``SPLITS`` partial softmax states of each ``(t, head group)``.

    Grid is ``(NUM_CORES,)``; each core walks ``(t, head group)`` round-robin.
    """
    cid = tl.program_id(0)
    N_HEAD_GROUPS: tl.constexpr = N1_C // BLOCK_H
    TOTAL_TASKS = T1 * N_HEAD_GROUPS

    offs_d = tl.arange(0, D_C)
    offs_bh = tl.arange(0, BLOCK_H)

    for task_idx in range(cid, TOTAL_TASKS, NUM_CORES):
        t = task_idx // N_HEAD_GROUPS
        hg = task_idx % N_HEAD_GROUPS
        ws_base = (t * N_HEAD_GROUPS + hg) * SPLITS

        acc = tl.load(
            ws_acc_ptr
            + ws_base * BLOCK_H * D_C
            + offs_bh[:, None] * D_C
            + offs_d[None, :]
        )
        m_i = tl.load(ws_m_ptr + ws_base * BLOCK_H + offs_bh)
        l_i = tl.load(ws_l_ptr + ws_base * BLOCK_H + offs_bh)

        for s in range(1, SPLITS):
            slot = ws_base + s
            acc_s = tl.load(
                ws_acc_ptr
                + slot * BLOCK_H * D_C
                + offs_bh[:, None] * D_C
                + offs_d[None, :]
            )
            m_s = tl.load(ws_m_ptr + slot * BLOCK_H + offs_bh)
            l_s = tl.load(ws_l_ptr + slot * BLOCK_H + offs_bh)

            m_new = tl.maximum(m_i, m_s)
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha_old = tl.exp(m_i - m_safe)
            alpha_old = tl.where(m_i == float("-inf"), 0.0, alpha_old)
            alpha_new = tl.exp(m_s - m_safe)
            alpha_new = tl.where(m_s == float("-inf"), 0.0, alpha_new)

            l_i = l_i * alpha_old + l_s * alpha_new
            acc = acc * alpha_old[:, None] + acc_s * alpha_new[:, None]
            m_i = m_new

        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        o = acc / safe_l[:, None]

        offs_h = hg * BLOCK_H + offs_bh
        tl.store(
            out_ptr + t * N1_C * D_C + offs_h[:, None] * D_C + offs_d[None, :],
            o.to(tl.bfloat16),
        )

        if RETURN_LSE:
            lse = m_i + tl.log(safe_l)
            tl.store(lse_ptr + t * N1_C + offs_h, lse)
