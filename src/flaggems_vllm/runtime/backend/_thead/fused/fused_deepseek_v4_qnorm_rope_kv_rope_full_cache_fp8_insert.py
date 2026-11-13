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

import os

import torch
import triton
import triton.language as tl

_DEFAULT_SPLIT_MIN_Q_ROWS = 8192


@triton.jit
def _f32_to_fp8_e4m3fn(x):
    b = x.to(tl.int32, bitcast=True)
    a = b & 0x7FFFFFFF
    t = a - 0x3C000000
    t += 0x0007FFFF + ((t >> 20) & 1)
    r_norm = t >> 20
    r_sub = (a.to(tl.float32, bitcast=True) * 512.0 + 8388608.0).to(
        tl.int32, bitcast=True
    ) - 0x4B000000
    r = tl.where(a >= 0x3C800000, r_norm, r_sub)
    return (r | ((b >> 24) & 0x80)).to(tl.uint8)


@triton.jit
def _thead_fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert_kernel(
    q_ptr,
    kv_ptr,
    slot_mapping_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    fp8_scale_ptr,
    q_fp8_scale_inv_ptr,
    q_fp8_ptr,
    k_cache_ptr,
    N,
    num_tokens_insert,
    eps,
    k_cache_stride0,
    k_cache_stride1,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    NO_ROPE_OFFSET: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    CACHE_BLOCK_IS_POW2: tl.constexpr,
    LOG2_CACHE_BLOCK_SIZE: tl.constexpr,
    APPLY_Q_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = tl.arange(0, ROWS_PER_BLOCK)
    rows = pid * ROWS_PER_BLOCK + row_offsets

    h1 = H + 1
    total = N * h1

    valid = rows < total
    if ROWS_PER_BLOCK <= h1:
        # At most one token boundary can occur inside this block. Keep the
        # division on the scalar block start and use only integer wrap logic
        # for the per-row token/slot vectors.
        row0 = pid * ROWS_PER_BLOCK
        token0 = row0 // h1
        slot0 = row0 - token0 * h1
        slot_linear = slot0 + row_offsets
        wrap = slot_linear >= h1
        token = token0 + wrap.to(tl.int32)
        slot = tl.where(wrap, slot_linear - h1, slot_linear)
    else:
        token = rows // h1
        slot = rows - token * h1

    is_q = slot < H
    q_row_mask = valid & is_q
    kv_row_mask = valid & (slot == H)

    tok_in_range = kv_row_mask & (token < num_tokens_insert)
    sm = tl.load(slot_mapping_ptr + token, mask=tok_in_range, other=-1)
    sm_ok = sm >= 0
    sm_safe = tl.where(sm_ok, sm, 0)
    pos = tl.load(position_ids_ptr + token, mask=valid, other=0).to(tl.int32)
    kv_scale = 1.0 / tl.load(fp8_scale_ptr).to(tl.float32)
    q_scale = tl.load(q_fp8_scale_inv_ptr).to(tl.float32)

    half_idx = tl.arange(0, HALF_ROPE_DIM)
    cos_base = cos_sin_cache_ptr + pos[:, None] * ROPE_DIM
    cos_v = tl.load(cos_base + half_idx[None, :], mask=valid[:, None], other=0.0)
    sin_v = tl.load(
        cos_base + half_idx[None, :] + HALF_ROPE_DIM,
        mask=valid[:, None],
        other=0.0,
    )

    if CACHE_BLOCK_IS_POW2:
        page = sm_safe >> LOG2_CACHE_BLOCK_SIZE
        page_off = sm_safe & (CACHE_BLOCK_SIZE - 1)
    else:
        page = sm_safe // CACHE_BLOCK_SIZE
        page_off = sm_safe - page * CACHE_BLOCK_SIZE
    k_base = page * k_cache_stride0 + page_off * k_cache_stride1

    offs_d = tl.arange(0, HEAD_DIM)
    nope_cols = offs_d < NO_ROPE_OFFSET

    # Per-row bases must be 64-bit: the largest supported Q tensors exceed
    # 2^31 elements. The in-row offsets remain 32-bit.
    q_base = (token * H + slot).to(tl.int64) * HEAD_DIM
    kv_base = token.to(tl.int64) * HEAD_DIM

    x_q = tl.load(
        q_ptr + q_base[:, None] + offs_d[None, :],
        mask=q_row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    x_kv = tl.load(
        kv_ptr + kv_base[:, None] + offs_d[None, :],
        mask=kv_row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    x = tl.where(is_q[:, None], x_q, x_kv)

    scale = tl.full((ROWS_PER_BLOCK,), 1.0, tl.float32)
    if APPLY_Q_NORM:
        mean_sq = tl.sum(x * x, axis=1) * (1.0 / HEAD_DIM)
        rstd = tl.rsqrt(mean_sq + eps)
        scale = tl.where(is_q, rstd, 1.0)

    row_scale = tl.where(is_q, q_scale, kv_scale)
    combined = scale * row_scale

    xq_f32 = x * combined[:, None]
    xq_f32 = tl.minimum(tl.maximum(xq_f32, -QUANT_MAX), QUANT_MAX)
    out_nope = _f32_to_fp8_e4m3fn(xq_f32)

    tl.store(
        q_fp8_ptr + q_base[:, None] + offs_d[None, :],
        out_nope,
        mask=q_row_mask[:, None] & nope_cols[None, :],
    )

    kv_store_mask = (kv_row_mask & sm_ok)[:, None]
    tl.store(
        k_cache_ptr + k_base[:, None] + offs_d[None, :],
        out_nope,
        mask=kv_store_mask & nope_cols[None, :],
    )

    rope_offs = tl.arange(0, ROPE_DIM)
    rope_dst = (NO_ROPE_OFFSET + rope_offs)[None, :]
    x_rope_q = tl.load(
        q_ptr + q_base[:, None] + rope_dst,
        mask=q_row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    x_rope_kv = tl.load(
        kv_ptr + kv_base[:, None] + rope_dst,
        mask=kv_row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    x_rope = tl.where(is_q[:, None], x_rope_q, x_rope_kv)

    xr = tl.reshape(x_rope, (ROWS_PER_BLOCK, HALF_ROPE_DIM, 2))
    even, odd = tl.split(xr)

    rot_e = even * cos_v - odd * sin_v
    rot_o = even * sin_v + odd * cos_v
    x_rope_rot = tl.reshape(tl.join(rot_e, rot_o), (ROWS_PER_BLOCK, ROPE_DIM))

    xr_f32 = x_rope_rot * combined[:, None]
    xr_f32 = tl.minimum(tl.maximum(xr_f32, -QUANT_MAX), QUANT_MAX)
    out_rope = _f32_to_fp8_e4m3fn(xr_f32)

    tl.store(
        q_fp8_ptr + q_base[:, None] + rope_dst,
        out_rope,
        mask=q_row_mask[:, None],
    )
    tl.store(
        k_cache_ptr + k_base[:, None] + rope_dst,
        out_rope,
        mask=kv_store_mask,
    )


@triton.jit
def _thead_fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert_kernel_q(
    q_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    q_fp8_scale_inv_ptr,
    q_fp8_ptr,
    N,
    eps,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    NO_ROPE_OFFSET: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    APPLY_Q_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = tl.arange(0, ROWS_PER_BLOCK)
    rows = pid * ROWS_PER_BLOCK + row_offsets
    total = N * H
    valid = rows < total

    # Q rows are flattened [token, head]. When H is a multiple of the row
    # tile, every row in a program belongs to one token and the grid has no
    # partial Q tile. Load one cos/sin pair and broadcast it to all Q rows.
    half_idx = tl.arange(0, HALF_ROPE_DIM)
    if H % ROWS_PER_BLOCK == 0:
        token0 = (pid * ROWS_PER_BLOCK) // H
        pos0 = tl.load(position_ids_ptr + token0).to(tl.int32)
        q_scale = tl.load(q_fp8_scale_inv_ptr).to(tl.float32)
        cos_base = cos_sin_cache_ptr + pos0 * ROPE_DIM
        cos_v = tl.load(cos_base + half_idx)
        sin_v = tl.load(cos_base + half_idx + HALF_ROPE_DIM)
    else:
        if ROWS_PER_BLOCK <= H:
            row0 = pid * ROWS_PER_BLOCK
            token0 = row0 // H
            head0 = row0 - token0 * H
            head_linear = head0 + row_offsets
            wrap = head_linear >= H
            token = token0 + wrap.to(tl.int32)
        else:
            token = rows // H

        pos = tl.load(position_ids_ptr + token, mask=valid, other=0).to(tl.int32)
        q_scale = tl.load(q_fp8_scale_inv_ptr).to(tl.float32)
        cos_base = cos_sin_cache_ptr + pos[:, None] * ROPE_DIM
        cos_v = tl.load(cos_base + half_idx[None, :], mask=valid[:, None], other=0.0)
        sin_v = tl.load(
            cos_base + half_idx[None, :] + HALF_ROPE_DIM,
            mask=valid[:, None],
            other=0.0,
        )

    offs_d = tl.arange(0, HEAD_DIM)
    nope_cols = offs_d < NO_ROPE_OFFSET
    q_base = rows.to(tl.int64) * HEAD_DIM

    x = tl.load(
        q_ptr + q_base[:, None] + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    scale = tl.full((ROWS_PER_BLOCK,), 1.0, tl.float32)
    if APPLY_Q_NORM:
        mean_sq = tl.sum(x * x, axis=1) * (1.0 / HEAD_DIM)
        scale = tl.rsqrt(mean_sq + eps)
    combined = scale * q_scale

    xq_f32 = x * combined[:, None]
    xq_f32 = tl.minimum(tl.maximum(xq_f32, -QUANT_MAX), QUANT_MAX)
    out_nope = _f32_to_fp8_e4m3fn(xq_f32)
    tl.store(
        q_fp8_ptr + q_base[:, None] + offs_d[None, :],
        out_nope,
        mask=valid[:, None] & nope_cols[None, :],
    )

    rope_offs = tl.arange(0, ROPE_DIM)
    rope_dst = (NO_ROPE_OFFSET + rope_offs)[None, :]
    x_rope = tl.load(
        q_ptr + q_base[:, None] + rope_dst,
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    xr = tl.reshape(x_rope, (ROWS_PER_BLOCK, HALF_ROPE_DIM, 2))
    even, odd = tl.split(xr)
    rot_e = even * cos_v - odd * sin_v
    rot_o = even * sin_v + odd * cos_v
    x_rope_rot = tl.reshape(tl.join(rot_e, rot_o), (ROWS_PER_BLOCK, ROPE_DIM))

    xr_f32 = x_rope_rot * combined[:, None]
    xr_f32 = tl.minimum(tl.maximum(xr_f32, -QUANT_MAX), QUANT_MAX)
    out_rope = _f32_to_fp8_e4m3fn(xr_f32)
    tl.store(
        q_fp8_ptr + q_base[:, None] + rope_dst,
        out_rope,
        mask=valid[:, None],
    )


@triton.jit
def _thead_fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert_kernel_kv(
    kv_ptr,
    slot_mapping_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    fp8_scale_ptr,
    k_cache_ptr,
    N,
    num_tokens_insert,
    k_cache_stride0,
    k_cache_stride1,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    NO_ROPE_OFFSET: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    CACHE_BLOCK_IS_POW2: tl.constexpr,
    LOG2_CACHE_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    valid = rows < N

    tok_in_range = valid & (rows < num_tokens_insert)
    sm = tl.load(slot_mapping_ptr + rows, mask=tok_in_range, other=-1)
    sm_ok = sm >= 0
    sm_safe = tl.where(sm_ok, sm, 0)
    pos = tl.load(position_ids_ptr + rows, mask=valid, other=0).to(tl.int32)
    kv_scale = 1.0 / tl.load(fp8_scale_ptr).to(tl.float32)

    half_idx = tl.arange(0, HALF_ROPE_DIM)
    cos_base = cos_sin_cache_ptr + pos[:, None] * ROPE_DIM
    cos_v = tl.load(cos_base + half_idx[None, :], mask=valid[:, None], other=0.0)
    sin_v = tl.load(
        cos_base + half_idx[None, :] + HALF_ROPE_DIM,
        mask=valid[:, None],
        other=0.0,
    )

    if CACHE_BLOCK_IS_POW2:
        page = sm_safe >> LOG2_CACHE_BLOCK_SIZE
        page_off = sm_safe & (CACHE_BLOCK_SIZE - 1)
    else:
        page = sm_safe // CACHE_BLOCK_SIZE
        page_off = sm_safe - page * CACHE_BLOCK_SIZE
    k_base = page * k_cache_stride0 + page_off * k_cache_stride1

    offs_d = tl.arange(0, HEAD_DIM)
    nope_cols = offs_d < NO_ROPE_OFFSET
    kv_base = rows.to(tl.int64) * HEAD_DIM

    x = tl.load(
        kv_ptr + kv_base[:, None] + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    xq_f32 = x * kv_scale
    xq_f32 = tl.minimum(tl.maximum(xq_f32, -QUANT_MAX), QUANT_MAX)
    out_nope = _f32_to_fp8_e4m3fn(xq_f32)
    kv_store_mask = (valid & sm_ok)[:, None]
    tl.store(
        k_cache_ptr + k_base[:, None] + offs_d[None, :],
        out_nope,
        mask=kv_store_mask & nope_cols[None, :],
    )

    rope_offs = tl.arange(0, ROPE_DIM)
    rope_dst = (NO_ROPE_OFFSET + rope_offs)[None, :]
    x_rope = tl.load(
        kv_ptr + kv_base[:, None] + rope_dst,
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    xr = tl.reshape(x_rope, (ROWS_PER_BLOCK, HALF_ROPE_DIM, 2))
    even, odd = tl.split(xr)
    rot_e = even * cos_v - odd * sin_v
    rot_o = even * sin_v + odd * cos_v
    x_rope_rot = tl.reshape(tl.join(rot_e, rot_o), (ROWS_PER_BLOCK, ROPE_DIM))

    xr_f32 = x_rope_rot * kv_scale
    xr_f32 = tl.minimum(tl.maximum(xr_f32, -QUANT_MAX), QUANT_MAX)
    out_rope = _f32_to_fp8_e4m3fn(xr_f32)
    tl.store(
        k_cache_ptr + k_base[:, None] + rope_dst,
        out_rope,
        mask=kv_store_mask,
    )


def fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
    q,
    kv,
    q_fp8,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    fp8_scale,
    q_fp8_scale_inv,
    eps,
    cache_block_size,
    apply_q_norm=True,
):
    head_dim = 512
    rope_dim = 64

    assert q.dim() == 3 and q.shape[2] == head_dim, "q must be [N, H, 512]"
    N, H, _ = q.shape
    assert (
        q.is_contiguous() and kv.is_contiguous() and q_fp8.is_contiguous()
    ), "q, kv and q_fp8 must be contiguous"
    assert q.dtype == kv.dtype and q.dtype in (
        torch.float16,
        torch.bfloat16,
    ), "q/kv must have the same float16 or bfloat16 dtype"
    assert (
        q_fp8.dtype == k_cache.dtype == torch.uint8
    ), "outputs must be uint8 containers for E4M3 FP8 bit patterns"
    assert (
        kv.shape == (N, head_dim) and q_fp8.shape == q.shape
    ), "kv must be [N, 512] and q_fp8 must match q"
    assert (
        isinstance(cache_block_size, int) and cache_block_size > 0
    ), "cache_block_size must be a positive int"
    assert (
        k_cache.dim() == 3
        and k_cache.shape[1] == cache_block_size
        and k_cache.shape[2] == head_dim
        and k_cache.stride(2) == 1
    ), "k_cache must be [num_blocks, cache_block_size, 512] with contiguous rows"

    for tensor in (slot_mapping, position_ids):
        assert (
            tensor.dim() == 1 and tensor.dtype == torch.int64
        ), "slot_mapping and position_ids must be 1D int64 tensors"
        assert tensor.is_contiguous(), "index tensors must be contiguous"
    assert (
        slot_mapping.numel() <= N and position_ids.numel() == N
    ), "positions must match N; slot_mapping may omit padded rows"
    assert (
        cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[1] == rope_dim
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.is_contiguous()
    ), "cos_sin_cache must be contiguous float32 [max_pos, 64]"
    for name, tensor in (
        ("fp8_scale", fp8_scale),
        ("q_fp8_scale_inv", q_fp8_scale_inv),
    ):
        assert (
            tensor.dtype == torch.float32 and tensor.numel() == 1
        ), f"{name} must contain one float32 value"

    # Row and RoPE-table offsets use int32 before the int64 row-base casts.
    assert N * (H + 1) < 2**31, "N * (H + 1) must fit int32 row indexing"
    assert (
        cos_sin_cache.shape[0] * rope_dim < 2**31
    ), "max_pos * rope_dim must fit int32 RoPE indexing"

    if N == 0:
        return

    half_rope_dim = rope_dim // 2
    no_rope_offset = head_dim - rope_dim
    quant_max = 448.0
    cache_block_size_int = int(cache_block_size)
    cache_block_is_pow2 = (cache_block_size_int & (cache_block_size_int - 1)) == 0
    log2_cache_block_size = (
        cache_block_size_int.bit_length() - 1 if cache_block_is_pow2 else 0
    )

    rows_per_block = int(os.getenv("DSV4_ROWS_PER_BLOCK", "16"))
    num_warps = int(os.getenv("DSV4_NUM_WARPS", "8"))
    split_min_q_rows = int(
        os.getenv("DSV4_SPLIT_MIN_Q_ROWS", str(_DEFAULT_SPLIT_MIN_Q_ROWS))
    )

    # H == 0 has no Q work and uses the split wrapper to launch KV only.
    if H == 0 or N * H >= split_min_q_rows:
        if H > 0:
            q_grid = (triton.cdiv(N * H, rows_per_block),)
            _thead_fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert_kernel_q[
                q_grid
            ](
                q,
                position_ids,
                cos_sin_cache,
                q_fp8_scale_inv,
                q_fp8,
                N,
                float(eps),
                H=H,
                HEAD_DIM=head_dim,
                ROPE_DIM=rope_dim,
                HALF_ROPE_DIM=half_rope_dim,
                NO_ROPE_OFFSET=no_rope_offset,
                QUANT_MAX=quant_max,
                ROWS_PER_BLOCK=rows_per_block,
                APPLY_Q_NORM=bool(apply_q_norm),
                num_warps=num_warps,
                num_stages=1,
            )

        kv_grid = (triton.cdiv(N, rows_per_block),)
        _thead_fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert_kernel_kv[
            kv_grid
        ](
            kv,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            fp8_scale,
            k_cache,
            N,
            int(slot_mapping.numel()),
            int(k_cache.stride(0)),
            int(k_cache.stride(1)),
            HEAD_DIM=head_dim,
            ROPE_DIM=rope_dim,
            HALF_ROPE_DIM=half_rope_dim,
            NO_ROPE_OFFSET=no_rope_offset,
            QUANT_MAX=quant_max,
            ROWS_PER_BLOCK=rows_per_block,
            CACHE_BLOCK_SIZE=cache_block_size_int,
            CACHE_BLOCK_IS_POW2=cache_block_is_pow2,
            LOG2_CACHE_BLOCK_SIZE=log2_cache_block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        return

    total_rows = N * (H + 1)
    grid = (triton.cdiv(total_rows, rows_per_block),)
    _thead_fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert_kernel[grid](
        q,
        kv,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        fp8_scale,
        q_fp8_scale_inv,
        q_fp8,
        k_cache,
        N,
        int(slot_mapping.numel()),
        float(eps),
        int(k_cache.stride(0)),
        int(k_cache.stride(1)),
        H=H,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        HALF_ROPE_DIM=half_rope_dim,
        NO_ROPE_OFFSET=no_rope_offset,
        QUANT_MAX=quant_max,
        ROWS_PER_BLOCK=rows_per_block,
        CACHE_BLOCK_SIZE=cache_block_size_int,
        CACHE_BLOCK_IS_POW2=cache_block_is_pow2,
        LOG2_CACHE_BLOCK_SIZE=log2_cache_block_size,
        APPLY_Q_NORM=bool(apply_q_norm),
        num_warps=num_warps,
        num_stages=1,
    )
