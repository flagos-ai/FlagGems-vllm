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

"""Layout views, geometry constants and argument checks shared by the kernels.

The ORI KV cache is a packed uint8 tensor in PA_BBND layout::

    ori_kv: [block_num, block_size, 1, 544] uint8

    row byte layout:  [0, 512)   fp8_e4m3 data, nope[448] || rope[64]
                      [512, 544) bf16 group scales, 16 groups x 32 elements

``ori_cache_views`` exposes the two interleaved fields as strided tensors over
the same storage, so the kernels can address them with their natural dtypes and
no copy is made.  The row pitch becomes 544 fp8 elements for the data view and
272 bf16 elements for the scale view.
"""

import math

import torch

__all__ = [
    "BLOCK_H",
    "BLOCK_S",
    "CMP_KV_ROW_BYTES",
    "D",
    "GATHER_CORES",
    "GROUP_SIZE",
    "KV_ROW_BYTES",
    "MAX_BLOCK_H",
    "MAX_SPLITK_BLOCK_H",
    "MIN_SPLIT_DEPTH",
    "MULTIBUFFER_OPTS",
    "NUM_CORES",
    "N_GROUPS",
    "ROW_STRIDE_DATA",
    "ROW_STRIDE_SCALE",
    "head_blocks",
    "log2_block_size",
    "ori_cache_views",
    "pick_block_h",
    "pick_splits",
    "validate_ori_sparse_args",
]

# Logical head dim: nope(448) || rope(64).  K == V under MLA absorb.
D = 512
GROUP_SIZE = 32
N_GROUPS = D // GROUP_SIZE  # 16
KV_ROW_BYTES = 544  # 512B fp8 data + 32B bf16 scale
CMP_KV_ROW_BYTES = 320  # fp4_e2m1 side, not implemented here

# Row pitch of the two strided views, in elements of their own dtype.
ROW_STRIDE_DATA = KV_ROW_BYTES  # fp8 is 1 byte
ROW_STRIDE_SCALE = KV_ROW_BYTES // 2  # bf16 is 2 bytes

# Default grid width for the matmul-carried kernels.  The 950DT has 32 Cube and
# 64 Vector cores, and the attention kernels are Cube-bound, so the Cube count
# is their useful upper bound: at T1=32 a 64-program attention grid measures the
# same as a 32-program one (103.0us against 102.9us).
NUM_CORES = 32

# The gather carries no matmul, so it wants every Vector core rather than the
# 32 Cube cores.  At T1=32 it scales nearly linearly up to 64: 212.1us at 32
# programs, 145.2us at 48, 108.7us at 64.  64 is the plateau, and what lies
# past it is a tail effect rather than bandwidth: 128, 192 and 256 programs all
# land at 107.9us, while 80 and 96 cost 165.1 and 140.7us, because the last
# partial wave of a non-multiple grid stalls the kernel.  Output is
# bit-identical at every width because the round-robin assignment covers each
# tile exactly once.
GATHER_CORES = 64

# Head group and sparse-token tile.  BLOCK_H is how many heads one program
# sweeps, and since all N1 heads share one KV tile (MLA absorb), it is also the
# re-read factor: at BLOCK_H=16 and N1=64 the attention reads every token's
# tile 4 times, once per head group.  BLOCK_H=64 reads it once and measures
# 49.9us against 102.9us at T1=32, so the dispatch takes the widest head block
# that still fills the grid.  64 is the widest that compiles here: the UB holds
# Q[64,512]bf16 64K plus the 128-token KV tile, with the fp32 accumulator
# carried in the Cube accumulator rather than UB.
BLOCK_H = 16
MAX_BLOCK_H = 64

# The split-K main kernel carries a [BLOCK_H, D] fp32 partial, so it runs out
# of UB one step earlier than the plain attention kernel: BLOCK_H=64 asks for
# 257.5 KB against 248 KB available.  The dispatch never pairs them, and an
# explicit request raises.
MAX_SPLITK_BLOCK_H = 32

# A split shallower than this measures worse than a narrower head block with
# fewer splits, even when both fill the grid: at T1=2 a 32-head block needs 8
# splits (256 tokens each) and costs 32.5us, while a 16-head block with 4
# splits (512 tokens each) costs 28.4us.  Depth feeds the k-loop; splits only
# buy programs.
MIN_SPLIT_DEPTH = 512

BLOCK_S = 64

# Auto double-buffering in the bisheng backend.  Without lifting the local
# buffer limits the gather and attention loops stall on MTE2 instead of
# overlapping the next tile's DMA with the current tile's math.
MULTIBUFFER_OPTS = {
    "multibuffer": True,
    "limit_auto_multi_buffer_only_for_local_buffer": False,
    "limit_auto_multi_buffer_of_local_buffer": "no-limit",
}


def ori_cache_views(ori_kv: torch.Tensor):
    """Return zero-copy ``(fp8 data, bf16 scale)`` views of a 544B/row KV cache.

    Args:
        ori_kv: [block_num, block_size, 1, 544] uint8, PA_BBND layout.

    Returns:
        data: [nrow, 512] float8_e4m3fn, stride (544, 1).
        scale: [nrow, 16] bfloat16, stride (272, 1), starting at byte 512.
    """
    storage = ori_kv.untyped_storage()
    nrow = ori_kv.shape[0] * ori_kv.shape[1]
    device = ori_kv.device

    data = torch.empty(0, dtype=torch.float8_e4m3fn, device=device)
    data.set_(storage, storage_offset=0, size=(nrow, D), stride=(ROW_STRIDE_DATA, 1))

    scale = torch.empty(0, dtype=torch.bfloat16, device=device)
    scale.set_(
        storage,
        storage_offset=D // 2,
        size=(nrow, N_GROUPS),
        stride=(ROW_STRIDE_SCALE, 1),
    )
    return data, scale


def log2_block_size(block_size: int) -> int:
    """Exponent of a power-of-two page size, so the kernels can shift and mask."""
    log2_bs = int(math.log2(block_size))
    if 1 << log2_bs != block_size:
        raise ValueError(f"block_size must be a power of 2, got {block_size}")
    return log2_bs


def head_blocks(n1: int, default: int = BLOCK_H, max_block_h: int = MAX_BLOCK_H):
    """Head block candidates for ``n1``, widest first.

    Every head sweeps the same KV tile, so a wider block reads the tile fewer
    times; the dispatch walks these in order and takes the first that keeps the
    grid full.
    """
    blocks = []
    candidate = min(n1, max_block_h)
    while candidate >= default:
        if n1 % candidate == 0:
            blocks.append(candidate)
        candidate //= 2
    return tuple(blocks) or (default,)


def pick_block_h(
    n1: int, default: int = BLOCK_H, max_block_h: int = MAX_BLOCK_H
) -> int:
    """Widest head block that divides ``n1``, capped at ``max_block_h``."""
    return head_blocks(n1, default, max_block_h)[0]


def pick_splits(num_cores: int, base_tasks: int, k_total: int, block_s: int) -> int:
    """Largest split count that fills the cores and tiles K evenly.

    A split must be a whole number of ``block_s`` tiles, otherwise the main
    kernel would need a masked tail per split.  Returns 1 when no such split
    exists, which the callers read as "do not split".
    """
    splits = max(1, num_cores // max(base_tasks, 1))
    splits = min(splits, k_total // block_s)
    while splits > 1:
        if k_total % splits == 0 and (k_total // splits) % block_s == 0:
            break
        splits -= 1
    return max(splits, 1)


def validate_ori_sparse_args(
    q,
    ori_kv,
    ori_sparse_indices,
    ori_block_table,
    ori_topk_length,
    sinks,
    block_h,
    block_s,
):
    """Check the ORI_SPARSE decode call form, raising before any launch.

    Shape and dtype violations raise ``ValueError``; call forms this op does not
    cover raise ``NotImplementedError`` so the caller keeps the vendor path
    instead of silently getting a different numerical contract.
    """
    if q.dtype != torch.bfloat16:
        raise ValueError(f"q must be bfloat16, got {q.dtype}")
    if q.dim() != 3:
        raise ValueError(f"q must be [T1, N1, D], got shape {tuple(q.shape)}")
    if q.shape[-1] != D:
        raise ValueError(f"head dim must be {D}, got {q.shape[-1]}")

    t1, n1 = q.shape[0], q.shape[1]
    if n1 % block_h != 0:
        raise ValueError(f"N1={n1} must be a multiple of block_h={block_h}")

    if ori_kv.dtype != torch.uint8:
        raise ValueError(f"ori_kv must be uint8, got {ori_kv.dtype}")
    if ori_kv.dim() != 4 or ori_kv.shape[2] != 1:
        raise ValueError(
            "ori_kv must be [block_num, block_size, 1, "
            f"{KV_ROW_BYTES}], got shape {tuple(ori_kv.shape)}"
        )
    if ori_kv.shape[3] != KV_ROW_BYTES:
        raise NotImplementedError(
            f"ori_kv row must be {KV_ROW_BYTES} bytes (quant_mode=1 on 950DT), "
            f"got {ori_kv.shape[3]}"
        )
    if not ori_kv.is_contiguous():
        raise ValueError("ori_kv must be contiguous")

    if ori_sparse_indices is None or ori_block_table is None:
        raise ValueError("ori_sparse_indices and ori_block_table are required")
    if ori_sparse_indices.dtype != torch.int32:
        raise ValueError(
            f"ori_sparse_indices must be int32, got {ori_sparse_indices.dtype}"
        )
    if ori_sparse_indices.dim() != 2 or ori_sparse_indices.shape[0] != t1:
        raise ValueError(
            f"ori_sparse_indices must be [T1={t1}, K], got "
            f"{tuple(ori_sparse_indices.shape)}"
        )
    k_total = ori_sparse_indices.shape[1]
    if k_total % block_s != 0:
        raise ValueError(f"K={k_total} must be a multiple of block_s={block_s}")

    if ori_block_table.dtype != torch.int32:
        raise ValueError(f"ori_block_table must be int32, got {ori_block_table.dtype}")
    if ori_block_table.dim() != 2 or ori_block_table.shape[0] != t1:
        raise ValueError(
            f"ori_block_table must be [T1={t1}, max_blocks], got "
            f"{tuple(ori_block_table.shape)}"
        )

    if ori_topk_length is not None:
        if ori_topk_length.dtype != torch.int32:
            raise ValueError(
                f"ori_topk_length must be int32, got {ori_topk_length.dtype}"
            )
        if ori_topk_length.numel() != t1:
            raise ValueError(
                f"ori_topk_length must hold T1={t1} entries, got "
                f"{ori_topk_length.numel()}"
            )

    if sinks is not None:
        # Casting here would be a torch compute op in a production path.
        if sinks.dtype != torch.float32:
            raise ValueError(f"sinks must be float32, got {sinks.dtype}")
        if sinks.numel() != n1:
            raise ValueError(f"sinks must hold N1={n1} entries, got {sinks.numel()}")

    log2_block_size(ori_kv.shape[1])
