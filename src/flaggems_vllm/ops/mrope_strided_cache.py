# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct-cache M-RoPE for strided vLLM PE views, with native output aliases.

The arithmetic statements follow vLLM 0.28 mrope._triton_mrope_forward.
Only addressing and output placement differ. The initial launch uses the exact
native fixed schedule (one token/program, four warps); see workflow-design.md.
No torch compute/copy/cast fallback exists in this module.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mrope_strided_cache_forward(
    q_ptr,
    k_ptr,
    cache,
    positions,
    q_out,
    k_out,
    cache_rows: tl.constexpr,
    q_row_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    k_row_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    k_out_row_stride: tl.constexpr,
    k_out_head_stride: tl.constexpr,
    pos_axis_stride: tl.constexpr,
    pos_token_stride: tl.constexpr,
    cache_row_stride: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    rd: tl.constexpr,
    pad_rd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride
    q_out = q_out + pid * (n_qh * rd)
    k_out = k_out + pid * k_out_row_stride
    half_rd = rd // 2

    # Preserve legal torch advanced-index semantics, including negative indices.
    t_position = tl.load(positions + pid * pos_token_stride)
    h_position = tl.load(positions + pos_axis_stride + pid * pos_token_stride)
    w_position = tl.load(positions + 2 * pos_axis_stride + pid * pos_token_stride)
    t_position = tl.where(t_position < 0, t_position + cache_rows, t_position)
    h_position = tl.where(h_position < 0, h_position + cache_rows, h_position)
    w_position = tl.where(w_position < 0, w_position + cache_rows, w_position)
    t_cos = cache + t_position * cache_row_stride
    h_cos = cache + h_position * cache_row_stride
    w_cos = cache + w_position * cache_row_stride
    t_sin = t_cos + half_rd
    h_sin = h_cos + half_rd
    w_sin = w_cos + half_rd

    # Exact native mask/addition expressions; do not replace with one selection
    # or change dtype/FP fusion merely because the real axes are disjoint.
    cos_offsets = tl.arange(0, pad_rd // 2)
    if is_interleaved:
        valid_mask = cos_offsets < half_rd
        h_mask = (
            valid_mask & ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        )
        w_mask = (
            valid_mask & ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        )
        t_mask = valid_mask & ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)
    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # Adjacent-pair branch copied from the native oracle. Both complete tiles
    # are loaded before an aliased K output can be written.
    rotary_offsets = tl.arange(0, pad_rd)
    q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * q_head_stride + rotary_offsets[None, :]
    )
    k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * k_head_stride + rotary_offsets[None, :]
    )
    q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (rotary_offsets[None, :] < rd)
    k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (rotary_offsets[None, :] < rd)
    q_tile = tl.load(q_ptr + q_offsets, mask=q_mask, other=0).to(sin_row.dtype)
    k_tile = tl.load(k_ptr + k_offsets, mask=k_mask, other=0).to(sin_row.dtype)
    q_tile_1, q_tile_2 = tl.split(tl.reshape(q_tile, (pad_n_qh, pad_rd // 2, 2)))
    k_tile_1, k_tile_2 = tl.split(tl.reshape(k_tile, (pad_n_kh, pad_rd // 2, 2)))

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    new_q_tile = tl.interleave(new_q_tile_1, new_q_tile_2)
    q_store_offsets = tl.arange(0, pad_n_qh)[:, None] * rd + rotary_offsets[None, :]
    tl.store(q_out + q_store_offsets, new_q_tile, mask=q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    new_k_tile = tl.interleave(new_k_tile_1, new_k_tile_2)
    k_store_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * k_out_head_stride + rotary_offsets[None, :]
    )
    tl.store(k_out + k_store_offsets, new_k_tile, mask=k_mask)


def _require(ok, message):
    if not ok:
        raise NotImplementedError(message)


def _metadata(tensor):
    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "lazy": tensor.is_conj() or tensor.is_neg(),
        "requires_grad": tensor.requires_grad,
    }


def validate_metadata(
    q, k, cache, positions, sections, interleaved, rotary_dim=64, neox=False
):
    """Pure metadata contract; also used by CPU tests without CUDA tensors."""
    _require(
        rotary_dim == 64 and neox is False, "only adjacent-pair rotary64 is supported"
    )
    _require(type(interleaved) is bool, "interleaved must be bool")
    _require(
        isinstance(sections, (tuple, list)) and len(sections) == 3,
        "three M-RoPE sections required",
    )
    _require(
        all(type(x) is int and x >= 0 for x in sections) and sum(sections) == 32,
        "nonnegative M-RoPE sections must sum to32",
    )
    _require(len(q["shape"]) == len(k["shape"]) == 3, "Q/K must be 3D PE views")
    rows, q_heads, width = q["shape"]
    _require(
        0 <= rows <= 1024 and q_heads in (4, 32) and width == 64,
        "Q outside initial T<=1024 H4/32 R64 scope",
    )
    _require(k["shape"] == (rows, 1, 64), "K must be [T,1,64]")
    _require(
        len(cache["shape"]) == 2 and cache["shape"][0] > 0 and cache["shape"][1] == 64,
        "cache must be nonempty [N,64]",
    )
    _require(positions["shape"] == (3, rows), "positions must be [3,T]")
    _require(
        q["device"].startswith("cuda")
        and all(x["device"] == q["device"] for x in (k, cache, positions)),
        "one CUDA device required",
    )
    _require(
        all(x["dtype"] == "torch.bfloat16" for x in (q, k, cache))
        and positions["dtype"] == "torch.int64",
        "BF16 Q/K/cache and INT64 positions required",
    )
    for value in (q, k, cache, positions):
        _require(
            value["layout"] == "torch.strided"
            and not value["lazy"]
            and not value["requires_grad"],
            "ordinary inference strided tensors required",
        )
        _require(
            all(type(s) is int and s > 0 for s in value["stride"]),
            "positive strides required",
        )
    for value in (q, k):
        _require(
            value["stride"][2] == 1 and value["stride"][1] >= 64,
            "unit PE stride and disjoint heads required",
        )
        span = (value["shape"][1] - 1) * value["stride"][1] + 64
        _require(
            rows <= 1 or value["stride"][0] >= span,
            "overlapping token rows unsupported",
        )
    _require(
        q["stride"][1] > 64,
        "initial scope requires gapped source Q heads so native reshape produces fresh Q",
    )
    _require(
        cache["stride"][1] == 1 and cache["stride"][0] >= 64,
        "unit cache channel stride and disjoint cache rows required",
    )
    _require(
        rows == 0 or positions["stride"][0] >= (rows - 1) * positions["stride"][1] + 1,
        "overlapping position axes unsupported",
    )
    return rows, q_heads


def can_mrope_strided_cache(
    query,
    key,
    cos_sin_cache,
    positions,
    sections,
    interleaved=True,
    rotary_dim=64,
    is_neox_style=False,
):
    """Capability gate with no allocation, synchronization or numerical work."""
    try:
        validate_metadata(
            *(_metadata(x) for x in (query, key, cos_sin_cache, positions)),
            sections,
            interleaved,
            rotary_dim,
            is_neox_style,
        )
        if query.shape[0]:
            pointers = [
                x.untyped_storage().data_ptr()
                for x in (query, key, cos_sin_cache, positions)
            ]
            _require(
                len(set(pointers)) == len(pointers), "shared input storage unsupported"
            )
    except (NotImplementedError, AttributeError, TypeError):
        return False
    return True


def mrope_strided_cache(
    query,
    key,
    cos_sin_cache,
    positions,
    sections=(12, 10, 10),
    interleaved=True,
    rotary_dim=64,
    is_neox_style=False,
):
    """Return native-compatible rotary PE outputs from strided input PE views.

    GPU position values must satisfy -cache_rows <= position < cache_rows.
    Every T row is computed, including padded rows. No valid-length shortcut,
    request cache, host value read or torch numerical fallback is used.
    """
    _require(
        can_mrope_strided_cache(
            query,
            key,
            cos_sin_cache,
            positions,
            sections,
            interleaved,
            rotary_dim,
            is_neox_style,
        ),
        "M-RoPE metadata/layout outside supported direct-cache scope",
    )
    rows, q_heads = query.shape[:2]
    q_out = torch.empty((rows, q_heads, 64), device=query.device, dtype=query.dtype)
    # This is metadata only. It mirrors the original 2D K contiguous() decision
    # without calling contiguous(), cloning or copying the tensor.
    k_alias = rows <= 1 or key.stride(0) == 64
    k_out = (
        key.view(rows, 64).view(rows, 1, 64)
        if k_alias
        else torch.empty((rows, 1, 64), device=key.device, dtype=key.dtype)
    )
    if rows:
        _mrope_strided_cache_forward[(rows,)](
            query,
            key,
            cos_sin_cache,
            positions,
            q_out,
            k_out,
            cos_sin_cache.shape[0],
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            k_out.stride(0),
            k_out.stride(1),
            positions.stride(0),
            positions.stride(1),
            cos_sin_cache.stride(0),
            q_heads,
            1,
            triton.next_power_of_2(q_heads),
            1,
            64,
            64,
            sections[0],
            sections[1],
            sections[2],
            interleaved,
            num_warps=4,
        )
    return q_out, k_out


__all__ = ["mrope_strided_cache", "can_mrope_strided_cache"]
