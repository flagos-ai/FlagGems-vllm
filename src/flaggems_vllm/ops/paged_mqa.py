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

"""FlagGems paged MQA adapted to read interleaved pages without repacking.

Derived from the FlagGems paged-MQA operator (Apache-2.0).
Native FP8 MMA keeps that computation. Other targets exactly decode E4M3FN
bytes to BF16 before MMA, preserving the page format and scale addressing.
"""

import triton
import triton.language as tl

from flaggems_vllm.ops.fp8_storage import decode_e4m3fn


@triton.jit
def _paged_mqa_logits_kernel(
    Q_ptr,  # [total_rows, H * D] uint8 (FP8 bitcast)
    KV_data_ptr,  # uint8 pages: [values | fp32 scales | optional padding]
    Weights_ptr,  # [total_rows, H] float32
    Block_tables_ptr,  # [total_rows, max_blocks_per_seq] int32
    Output_ptr,  # [total_rows, max_model_len] float32
    Ctx_lens_ptr,  # [total_rows] int32
    total_rows,
    max_ctx,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    max_model_len,
    block_size: tl.constexpr,
    max_blocks_per_seq,
    num_phys_blocks,
    stride_q_row,
    stride_kv_page,
    stride_bt_row,
    stride_out_row,
    stride_w_row,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
    NEXT_N: tl.constexpr,
    CTX_PER_ROW: tl.constexpr,
    BT_PER_ROW: tl.constexpr,
):
    """Per-tile kernel: each program processes one BLOCK_KV tile for one row."""
    kv_block = tl.program_id(0)
    row_idx = tl.program_id(1)

    if row_idx >= total_rows:
        return

    ctx_len = tl.load(Ctx_lens_ptr + (row_idx if CTX_PER_ROW else row_idx // NEXT_N))
    kv_start = kv_block * BLOCK_KV
    if kv_start >= ctx_len:
        return

    q_row_base = Q_ptr + row_idx * stride_q_row
    w_row_base = Weights_ptr + row_idx * stride_w_row
    bt_row_base = (
        Block_tables_ptr
        + (row_idx if BT_PER_ROW else row_idx // NEXT_N) * stride_bt_row
    )
    out_row_base = Output_ptr + row_idx * stride_out_row

    h_ids = tl.arange(0, num_heads)
    d_ids = tl.arange(0, BLOCK_D)

    # Pre-load Q as FP8: [num_heads, head_dim]
    q_offsets = h_ids[:, None] * head_dim + d_ids[None, :]
    q_u8 = tl.load(q_row_base + q_offsets)
    if NATIVE_FP8:
        q_values = q_u8.to(tl.float8e4nv, bitcast=True)
    else:
        # Every finite E4M3FN value is exactly representable in BF16.
        q_values = decode_e4m3fn(q_u8).to(tl.bfloat16)

    # Pre-load weights: [num_heads] float32
    w_all = tl.load(w_row_base + tl.arange(0, num_heads))

    end_pos = tl.minimum(kv_start + BLOCK_KV, ctx_len)
    first_lb = kv_start // block_size

    p_ids = tl.arange(0, block_size)

    for blk_idx in range(NUM_BLOCKS):
        lb = first_lb + blk_idx
        logical_base = lb * block_size
        if logical_base < end_pos:
            # Block-table lookup for physical block index
            phys_block = tl.load(bt_row_base + lb)
            phys_block = tl.maximum(phys_block, 0)
            phys_block = tl.minimum(phys_block, num_phys_blocks - 1)
            page_base = KV_data_ptr + phys_block * stride_kv_page

            # Coalesced KV load: [block_size, head_dim] as FP8
            kv_offsets = p_ids[:, None] * head_dim + d_ids[None, :]
            kv_u8 = tl.load(page_base + kv_offsets)
            if NATIVE_FP8:
                kv_values = kv_u8.to(tl.float8e4nv, bitcast=True)
            else:
                kv_values = decode_e4m3fn(kv_u8).to(tl.bfloat16)

            # Tensor-core MMA: Q[H, D] @ KV[block_size, D]^T -> [H, block_size]
            dots = tl.dot(q_values, tl.trans(kv_values))

            # Coalesced scale load: [block_size] float32
            scale_tile = tl.load(
                (page_base + block_size * head_dim).to(tl.pointer_type(tl.float32))
                + p_ids
            )

            # Fused scale, relu, weight, reduce over heads
            scores = tl.maximum(dots * scale_tile[None, :], 0.0)
            weighted = scores * w_all[:, None]
            output_tile = tl.sum(weighted, axis=0)

            pos_ids = logical_base + p_ids
            valid_mask = pos_ids < end_pos
            tl.store(out_row_base + pos_ids, output_tile, mask=valid_mask)


def paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
):
    """Static-grid E4M3FN MQA over physical pages, including padded page strides.

    Query scales must already be folded into weights. No context length is read
    on the host and the original KV allocation is never repacked.
    """
    import torch

    from flaggems_vllm import runtime
    from flaggems_vllm.ops.data_movement import fill

    del schedule_metadata
    values, scale = q
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if values.ndim != 4 or kv_cache.ndim != 4:
        raise ValueError("Paged MQA expects rank-4 query and cache")
    batch, next_n, heads, dim = values.shape
    rows = batch * next_n
    page = kv_cache.shape[1]
    if (
        dim != 128
        or heads not in (16, 32, 64)
        or page not in (32, 64)
        or values.dtype != torch.float8_e4m3fn
        or scale is not None
        or kv_cache.dtype != torch.uint8
        or kv_cache.shape[2:] != (1, dim + 4)
        or kv_cache.stride(-1) != 1
        or kv_cache.stride(1) != dim + 4
        or kv_cache.stride(0) < page * (dim + 4)
        or kv_cache.stride(0) % 4
        or not values.is_contiguous()
    ):
        raise ValueError("Unsupported paged MQA FP8 page/query layout")
    if (
        weights.shape != (rows, heads)
        or weights.dtype != torch.float32
        or not weights.is_contiguous()
    ):
        raise ValueError("Paged MQA weights must be contiguous float32 [rows, heads]")
    if (
        context_lens.shape not in ((batch,), (batch, next_n))
        or context_lens.dtype != torch.int32
        or not context_lens.is_contiguous()
    ):
        raise ValueError(
            "Paged MQA context lengths must be contiguous int32 [B] or [B,next_n]"
        )
    if (
        block_tables.ndim != 2
        or block_tables.shape[0] not in (batch, rows)
        or block_tables.dtype != torch.int32
        or block_tables.stride(1) != 1
    ):
        raise ValueError("Paged MQA block tables must be int32 [B or rows, pages]")
    if max_model_len < 0 or block_tables.shape[1] * page < max_model_len:
        raise ValueError("Paged MQA block table cannot address configured model length")
    if any(
        t.device != values.device
        for t in (kv_cache, weights, context_lens, block_tables)
    ):
        raise ValueError("Paged MQA tensors must share the query device")
    logits = torch.empty(
        (rows, max_model_len), device=values.device, dtype=torch.float32
    )
    fill(logits, -float("inf") if clean_logits else 0.0)
    if rows == 0 or max_model_len == 0:
        return logits
    native_fp8 = (
        runtime.device.vendor_name == "nvidia"
        and torch.cuda.get_device_capability(values.device) >= (8, 9)
    )
    # Tile covers complete physical pages. This is the original paged-MQA
    # selection, now owned by the library instead of a private framework peek.
    tile = (
        256
        if max_model_len <= 2048
        else 512 if max_model_len <= 4096 else 1024 if max_model_len <= 8192 else 2048
    )
    _paged_mqa_logits_kernel[(triton.cdiv(max_model_len, tile), rows)](
        values.view(torch.uint8),
        kv_cache,
        weights,
        block_tables,
        logits,
        context_lens,
        rows,
        max_model_len,
        heads,
        dim,
        max_model_len,
        page,
        block_tables.shape[1],
        kv_cache.shape[0],
        heads * dim,
        kv_cache.stride(0),
        block_tables.stride(0),
        max_model_len,
        heads,
        BLOCK_KV=tile,
        BLOCK_D=128,
        NUM_BLOCKS=tile // page,
        NATIVE_FP8=native_fp8,
        NEXT_N=next_n,
        CTX_PER_ROW=context_lens.ndim == 2,
        BT_PER_ROW=block_tables.shape[0] == rows,
    )
    return logits
