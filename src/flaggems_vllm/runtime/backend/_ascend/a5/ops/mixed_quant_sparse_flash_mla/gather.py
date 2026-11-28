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

"""Gather + dequant pass: scattered FP8 pages to a contiguous bf16 buffer.

Resolves ``sparse_indices -> block_table -> physical row``, loads the FP8 row
and its 16 bf16 group scales, dequantizes to bf16 and writes a dense
``[T1, K, D]`` buffer plus a ``[T1, K]`` validity mask.

Splitting the gather out of the attention is what makes the multi-token cases
fast: the attention pass then reads contiguous rows instead of issuing one
random 544B descriptor per token for every head group, so each page is fetched
from HBM once per query instead of once per head group.  This pass is 68% of
kernel time at T1=32 (108.8us of 159.7us).

It is bound on the number of HBM granule accesses, not on bytes or locality.
A 544B row straddles two 512B granules, so each gathered token costs two
accesses: the 512B data load and the 32B scale load measure 58.3us each at
T1=32, half the total, despite a 16-fold difference in bytes.  Merging them
into one 1024B load is still two granules and costs more, sorting or
clustering the indices changes nothing, and the grid plateaus at 64 programs.
``launcher.py`` carries those measurements.
"""

import triton
import triton.language as tl

__all__ = ["gather_dequant_kernel"]


@triton.jit
def gather_dequant_kernel(
    # ── Inputs ──
    ori_data_ptr,  # fp8_e4m3 [nrow, D_C], row pitch ROW_STRIDE_DATA
    ori_scale_ptr,  # bf16 [nrow, NG], row pitch ROW_STRIDE_SCALE
    ori_idx_ptr,  # [T1, K_TOTAL] int32
    ori_bt_ptr,  # [T1, MAX_BLK] int32
    ori_topk_ptr,  # [T1] int32, or a dummy when HAS_TOPK is False
    # ── Outputs ──
    deq_out_ptr,  # [T1, K_TOTAL, D_C] bf16
    valid_out_ptr,  # [T1, K_TOTAL] int8
    # ── Constexprs ──
    T1: tl.constexpr,
    K_TOTAL: tl.constexpr,
    MAX_BLK: tl.constexpr,
    BS: tl.constexpr,
    LOG2_BS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NUM_CORES: tl.constexpr,
    D_C: tl.constexpr,
    NG: tl.constexpr,
    GS: tl.constexpr,
    ROW_STRIDE_DATA: tl.constexpr,
    ROW_STRIDE_SCALE: tl.constexpr,
    HAS_TOPK: tl.constexpr,
):
    """Gather sparse KV rows, dequantize FP8 to bf16, write contiguously.

    Grid is ``(NUM_CORES,)``; each core walks ``(t, s_tile)`` tasks round-robin.
    """
    cid = tl.program_id(0)
    N_S_TILES: tl.constexpr = K_TOTAL // BLOCK_S
    TOTAL_TASKS = T1 * N_S_TILES

    offs_d = tl.arange(0, D_C)
    offs_sc = tl.arange(0, NG)
    offs_tile = tl.arange(0, BLOCK_S)

    for task_idx in range(cid, TOTAL_TASKS, NUM_CORES):
        t = task_idx // N_S_TILES
        tile_i = task_idx % N_S_TILES
        s0 = tile_i * BLOCK_S
        offs_s = s0 + offs_tile

        idx = tl.load(ori_idx_ptr + t * K_TOTAL + offs_s)
        if HAS_TOPK:
            valid_cnt = tl.load(ori_topk_ptr + t).to(tl.int32)
            valid = (idx >= 0) & (offs_s < valid_cnt)
        else:
            valid = idx >= 0
        # Clamp instead of masking the load: an out-of-range index would make
        # the block-table lookup read past MAX_BLK.
        idxc = tl.maximum(idx, 0)

        # PA_BBND address resolution.
        logical_blk = idxc >> LOG2_BS
        off_in_blk = idxc & (BS - 1)
        phys_blk = tl.load(ori_bt_ptr + t * MAX_BLK + logical_blk)
        row = phys_blk * BS + off_in_blk

        k_fp8 = tl.load(ori_data_ptr + row[:, None] * ROW_STRIDE_DATA + offs_d[None, :])
        sc = tl.load(ori_scale_ptr + row[:, None] * ROW_STRIDE_SCALE + offs_sc[None, :])

        # Dequantize: one bf16 scale per group of GS channels.
        k3 = tl.reshape(k_fp8.to(tl.bfloat16), (BLOCK_S, NG, GS))
        k_deq = tl.reshape(k3 * sc.to(tl.bfloat16)[:, :, None], (BLOCK_S, D_C))

        out_base = t * K_TOTAL * D_C + s0 * D_C
        tl.store(
            deq_out_ptr + out_base + offs_tile[:, None] * D_C + offs_d[None, :],
            k_deq,
        )
        tl.store(valid_out_ptr + t * K_TOTAL + offs_s, valid.to(tl.int8))
