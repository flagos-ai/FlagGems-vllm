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

from flaggems_vllm.runtime.backend._enflame.ops.fp8_storage import decode_e4m3fn


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
    fill_value,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
    NEXT_N: tl.constexpr,
    CTX_PER_ROW: tl.constexpr,
    BT_PER_ROW: tl.constexpr,
):
    """Per-tile kernel: each program processes one BLOCK_KV tile for one row.

    Programs also write the fill value (-inf or 0) for every position at or
    beyond the row's context length, so the output needs no separate fill.

    Two constexpr structural arms (2026-11-12 RSI round): tiles covering a
    single page use the straight-line ordered body; multi-page tiles use a
    branch-free, mask-free loop over the pages fully inside the context, an
    epilogue for the at-most-one partially covered page, and one guarded
    fill store for the tail. The full-page loop drops the per-page
    ``tl.where``/store-mask tax that dominated multi-page tiles (~-7~-10%).
    """
    kv_block = tl.program_id(0)
    row_idx = tl.program_id(1)

    if row_idx >= total_rows:
        return

    ctx_len = tl.load(Ctx_lens_ptr + (row_idx if CTX_PER_ROW else row_idx // NEXT_N))
    # Prologue: Q/weight loads are independent of ctx_len, issue them before
    # the early-exit branch so their latency overlaps the ctx_len load.
    q_row_base = Q_ptr + row_idx * stride_q_row
    w_row_base = Weights_ptr + row_idx * stride_w_row
    h_ids = tl.arange(0, num_heads)
    d_ids = tl.arange(0, BLOCK_D)
    q_offsets = h_ids[:, None] * head_dim + d_ids[None, :]
    q_u8 = tl.load(q_row_base + q_offsets)
    if NATIVE_FP8:
        q_values = q_u8.to(tl.float8e4nv, bitcast=True)
    else:
        # Every finite E4M3FN value is exactly representable in BF16.
        q_values = decode_e4m3fn(q_u8).to(tl.bfloat16)
    w_all = tl.load(w_row_base + tl.arange(0, num_heads))

    kv_start = kv_block * BLOCK_KV
    tile_pos = kv_start + tl.arange(0, BLOCK_KV)
    out_row_base = Output_ptr + row_idx * stride_out_row
    if kv_start >= ctx_len:
        fill_tile = tl.zeros([BLOCK_KV], dtype=tl.float32) + fill_value
        tl.store(
            out_row_base + tile_pos,
            fill_tile,
            mask=tile_pos < max_model_len,
        )
        return

    bt_row_base = (
        Block_tables_ptr
        + (row_idx if BT_PER_ROW else row_idx // NEXT_N) * stride_bt_row
    )

    end_pos = tl.minimum(kv_start + BLOCK_KV, ctx_len)
    first_lb = kv_start // block_size
    p_ids = tl.arange(0, block_size)

    if NUM_BLOCKS == 1:
        # Single-page arm: ordered loads, one page, masked store semantics.
        logical_base = kv_start
        phys_block = tl.load(bt_row_base + first_lb)
        phys_block = tl.maximum(phys_block, 0)
        phys_block = tl.minimum(phys_block, num_phys_blocks - 1)
        page_base = KV_data_ptr + phys_block * stride_kv_page

        kv_offsets = p_ids[:, None] * head_dim + d_ids[None, :]
        kv_u8 = tl.load(page_base + kv_offsets)
        if NATIVE_FP8:
            kv_values = kv_u8.to(tl.float8e4nv, bitcast=True)
        else:
            kv_values = decode_e4m3fn(kv_u8).to(tl.bfloat16)

        scale_tile = tl.load(
            (page_base + block_size * head_dim).to(tl.pointer_type(tl.float32)) + p_ids
        )

        dots = tl.dot(q_values, tl.trans(kv_values))

        scores = tl.maximum(dots * scale_tile[None, :], 0.0)
        weighted = scores * w_all[:, None]
        output_tile = tl.sum(weighted, axis=0)

        pos_ids = logical_base + p_ids
        valid_mask = pos_ids < end_pos
        fill_page = tl.zeros([block_size], dtype=tl.float32) + fill_value
        out_vals = tl.where(valid_mask, output_tile, fill_page)
        tl.store(out_row_base + pos_ids, out_vals, mask=pos_ids < max_model_len)
    else:
        # Multi-page arm: pages fully inside the context run branch-free and
        # mask-free; their stores never exceed end_pos <= max_model_len.
        span = end_pos - kv_start
        n_full = tl.minimum(span // block_size, NUM_BLOCKS)
        for blk_idx in range(n_full):
            lb = first_lb + blk_idx
            logical_base = lb * block_size
            phys_block = tl.load(bt_row_base + lb)
            phys_block = tl.maximum(phys_block, 0)
            phys_block = tl.minimum(phys_block, num_phys_blocks - 1)
            page_base = KV_data_ptr + phys_block * stride_kv_page

            kv_offsets = p_ids[:, None] * head_dim + d_ids[None, :]
            kv_u8 = tl.load(page_base + kv_offsets)
            if NATIVE_FP8:
                kv_values = kv_u8.to(tl.float8e4nv, bitcast=True)
            else:
                kv_values = decode_e4m3fn(kv_u8).to(tl.bfloat16)

            scale_tile = tl.load(
                (page_base + block_size * head_dim).to(tl.pointer_type(tl.float32))
                + p_ids
            )

            dots = tl.dot(q_values, tl.trans(kv_values))

            scores = tl.maximum(dots * scale_tile[None, :], 0.0)
            weighted = scores * w_all[:, None]
            output_tile = tl.sum(weighted, axis=0)

            tl.store(out_row_base + logical_base + p_ids, output_tile)

        # At most one partially covered page.
        partial_base = kv_start + n_full * block_size
        if partial_base < end_pos:
            lb = first_lb + n_full
            logical_base = lb * block_size
            phys_block = tl.load(bt_row_base + lb)
            phys_block = tl.maximum(phys_block, 0)
            phys_block = tl.minimum(phys_block, num_phys_blocks - 1)
            page_base = KV_data_ptr + phys_block * stride_kv_page

            kv_offsets = p_ids[:, None] * head_dim + d_ids[None, :]
            kv_u8 = tl.load(page_base + kv_offsets)
            if NATIVE_FP8:
                kv_values = kv_u8.to(tl.float8e4nv, bitcast=True)
            else:
                kv_values = decode_e4m3fn(kv_u8).to(tl.bfloat16)

            scale_tile = tl.load(
                (page_base + block_size * head_dim).to(tl.pointer_type(tl.float32))
                + p_ids
            )

            dots = tl.dot(q_values, tl.trans(kv_values))

            scores = tl.maximum(dots * scale_tile[None, :], 0.0)
            weighted = scores * w_all[:, None]
            output_tile = tl.sum(weighted, axis=0)

            pos_ids = logical_base + p_ids
            valid_mask = pos_ids < end_pos
            fill_page = tl.zeros([block_size], dtype=tl.float32) + fill_value
            out_vals = tl.where(valid_mask, output_tile, fill_page)
            tl.store(out_row_base + pos_ids, out_vals, mask=pos_ids < max_model_len)

        # Single guarded fill store for everything past the partial page.
        if end_pos < kv_start + BLOCK_KV:
            has_partial = partial_base < end_pos
            fill_begin = partial_base + tl.where(has_partial, block_size, 0)
            fill_pos = fill_begin + tl.arange(0, BLOCK_KV)
            fill_vals = tl.zeros([BLOCK_KV], dtype=tl.float32) + fill_value
            tile_end = kv_start + BLOCK_KV
            tl.store(
                out_row_base + fill_pos,
                fill_vals,
                mask=(fill_pos < max_model_len) & (fill_pos < tile_end),
            )


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
    if rows == 0 or max_model_len == 0:
        return logits
    native_fp8 = runtime.device.vendor_name == "enflame" or (
        runtime.device.vendor_name == "nvidia"
        and torch.cuda.get_device_capability(values.device) >= (8, 9)
    )
    # Tile covers complete physical pages. Tile size follows a GCU wave
    # model: with WAVE programs resident per wave, the makespan of one row
    # tile is ceil(total_pages / P / WAVE) * (PROLOGUE_US + P * PAGE_US),
    # where PROLOGUE_US/PAGE_US were calibrated on L600 (num_warps=1) and
    # cover the Q-row reload plus the per-page gather/MMA/reduce cost. Pick
    # the power-of-two page count P minimizing that model, with a 64-token
    # floor keeping the MMA tile wide enough for page_size 32 caches.
    total_pages = rows * triton.cdiv(max_model_len, page)
    pages_per_prog = 1
    best_cost = None
    for candidate in (1, 2, 4, 8, 16, 32):
        waves = triton.cdiv(triton.cdiv(total_pages, candidate), 48)
        cost = waves * (4.5 + candidate * 2.8)
        if best_cost is None or cost < best_cost:
            pages_per_prog = candidate
            best_cost = cost
    tile = max(64, pages_per_prog * page)
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
        -float("inf") if clean_logits else 0.0,
        num_warps=1,
        BLOCK_KV=tile,
        BLOCK_D=128,
        NUM_BLOCKS=tile // page,
        NATIVE_FP8=native_fp8,
        NEXT_N=next_n,
        CTX_PER_ROW=context_lens.ndim == 2,
        BT_PER_ROW=block_tables.shape[0] == rows,
    )
    return logits
