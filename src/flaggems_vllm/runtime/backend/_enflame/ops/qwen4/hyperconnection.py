# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe Triton kernels for Qwen4 gated HyperConnection.

This module contains the Qwen4 HC kernels, including delayed combine + norm.
The model wiring and vLLM custom-op registration stay in the model/plugin
repository.  The wrappers below are standalone FlagGems-vllm entry points and
fail closed when their accelerator/layout contract is not met.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HAS_TRITON = True


def can_use_hc_triton(*tensors: torch.Tensor) -> bool:
    """Return whether the specialized contiguous inference path is usable."""

    return bool(
        HAS_TRITON
        and tensors
        and len({tensor.device for tensor in tensors}) == 1
        and all(
            tensor.device.type not in ("cpu", "meta")
            and tensor.dtype in (torch.bfloat16, torch.float16)
            and tensor.is_contiguous()
            for tensor in tensors
        )
    )


def _has_flat_row_layout(tensor: torch.Tensor) -> bool:
    """Return whether leading dims can be flattened using ``stride(-2)``.

    Packed HC projection output is split into ``[..., lowrank]`` and
    ``[..., hc_count]`` views.  The latter is contiguous within each row but
    retains the packed projection's wider row stride.  The injection kernel
    accepts that layout explicitly, while still rejecting transposes and
    layouts whose leading dimensions cannot be flattened into rows.
    """

    if tensor.ndim < 2 or tensor.stride(-1) != 1:
        return False
    if tensor.stride(-2) < tensor.shape[-1]:
        return False
    return all(
        tensor.stride(dim) == tensor.shape[dim + 1] * tensor.stride(dim + 1)
        for dim in range(tensor.ndim - 2)
    )


def can_use_hc_inject_triton(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
) -> bool:
    """Allow packed-projection logits with a wider row stride.

    ``_hc_inject_combine_kernel`` consumes ``stride_logits_row`` and only
    requires the branch dimension to be contiguous.  The other two inputs
    retain the stricter contiguous inference contract.
    """

    tensors = (injection_logits, block_output, residual)
    return bool(
        HAS_TRITON
        and all(
            tensor.device.type not in ("cpu", "meta")
            and tensor.dtype in (torch.bfloat16, torch.float16)
            for tensor in tensors
        )
        and len({tensor.device for tensor in tensors}) == 1
        and _has_flat_row_layout(injection_logits)
        and block_output.is_contiguous()
        and residual.is_contiguous()
    )


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    group_row = tl.program_id(0)
    offsets = tl.arange(0, block_h)
    mask = offsets < hidden_size
    input_base = group_row * hidden_size
    weight_base = (group_row % hc_count) * hidden_size
    values = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    weight = tl.load(weight_ptr + weight_base + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / hidden_size + eps)
    tl.store(
        output_ptr + input_base + offsets,
        values * inv_rms * (1.0 + weight),
        mask=mask,
    )


@triton.jit
def _hc_gate_reduce_kernel(
    logits_ptr,
    normed_ptr,
    output_ptr,
    stride_logits_row,
    stride_normed_row,
    stride_output_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.program_id(1) * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    accumulator = tl.zeros((block_h,), dtype=tl.float32)
    for branch in tl.static_range(0, hc_count):
        branch_offsets = branch * hidden_size + offsets
        logits = tl.load(
            logits_ptr + row * stride_logits_row + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        normed = tl.load(
            normed_ptr + row * stride_normed_row + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sigmoid(logits) * normed
    tl.store(
        output_ptr + row * stride_output_row + offsets,
        accumulator / hc_count,
        mask=mask,
    )


@triton.jit
def _hc_inject_combine_kernel(
    injection_logits_ptr,
    block_output_ptr,
    residual_ptr,
    output_ptr,
    stride_logits_row,
    stride_block_row,
    stride_residual_row,
    stride_output_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    branch = tl.program_id(1)
    offsets = tl.program_id(2) * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    branch_offsets = branch * hidden_size + offsets
    logits = tl.load(injection_logits_ptr + row * stride_logits_row + branch).to(
        tl.float32
    )
    injection_weight = 2.0 * tl.sigmoid(logits / hc_count)
    block_output = tl.load(
        block_output_ptr + row * stride_block_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * stride_residual_row + branch_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + branch_offsets,
        residual + block_output * injection_weight,
        mask=mask,
    )


def qwen4_grouped_gemma_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> torch.Tensor:
    if hidden_states.ndim < 2 or hidden_states.shape[-1] != weight.numel():
        raise ValueError("Qwen4 HC RMSNorm received incompatible input and weight")
    if hc_count <= 0 or weight.numel() % hc_count or eps <= 0:
        raise ValueError("Qwen4 HC RMSNorm requires a valid HC count")
    if not can_use_hc_triton(hidden_states, weight):
        raise RuntimeError("Qwen4 HC RMSNorm requires contiguous accelerator tensors")
    output = torch.empty_like(hidden_states)
    if not hidden_states.numel():
        return output
    hidden_size = weight.numel() // hc_count
    block_h = triton.next_power_of_2(hidden_size)
    rows = hidden_states.numel() // weight.numel()
    _grouped_gemma_rmsnorm_kernel[(rows * hc_count,)](
        hidden_states,
        weight,
        output,
        hidden_size=hidden_size,
        hc_count=hc_count,
        eps=eps,
        block_h=block_h,
        num_warps=8 if block_h > 2048 else 4,
    )
    return output


def qwen4_grouped_gemma_rmsnorm_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> torch.Tensor:
    del weight, hc_count, eps
    return torch.empty_like(hidden_states)


def qwen4_hc_gate_reduce(
    logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if logits.shape != normed.shape or logits.ndim < 2:
        raise ValueError("Qwen4 HC gate logits and normalized input must match")
    if hc_count <= 0 or logits.shape[-1] % hc_count:
        raise ValueError("Qwen4 HC gate reduction requires a valid HC count")
    if not can_use_hc_triton(logits, normed):
        raise RuntimeError("Qwen4 HC gate reduction requires contiguous tensors")
    hidden_size = logits.shape[-1] // hc_count
    output = torch.empty(
        (*logits.shape[:-1], hidden_size), dtype=normed.dtype, device=normed.device
    )
    if not logits.numel():
        return output
    rows = logits.numel() // logits.shape[-1]
    block_h = 256
    _hc_gate_reduce_kernel[(rows, triton.cdiv(hidden_size, block_h))](
        logits,
        normed,
        output,
        logits.stride(-2),
        normed.stride(-2),
        output.stride(-2),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_h=block_h,
        num_warps=4,
    )
    return output


def qwen4_hc_gate_reduce_fake(
    logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del normed
    return torch.empty(
        (*logits.shape[:-1], logits.shape[-1] // hc_count),
        dtype=logits.dtype,
        device=logits.device,
    )


def qwen4_hc_inject_combine(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if injection_logits.shape != (*block_output.shape[:-1], hc_count):
        raise ValueError("Qwen4 HC injection logits have an invalid shape")
    if hc_count <= 0:
        raise ValueError("Qwen4 HC injection requires a positive HC count")
    if residual.shape != (
        *block_output.shape[:-1],
        hc_count * block_output.shape[-1],
    ):
        raise ValueError("Qwen4 HC residual and block output shapes are incompatible")
    if not can_use_hc_inject_triton(injection_logits, block_output, residual):
        raise RuntimeError(
            "Qwen4 HC injection received an unsupported accelerator layout"
        )
    output = torch.empty_like(residual)
    if not residual.numel():
        return output
    hidden_size = block_output.shape[-1]
    rows = block_output.numel() // hidden_size
    block_h = 256
    _hc_inject_combine_kernel[(rows, hc_count, triton.cdiv(hidden_size, block_h))](
        injection_logits,
        block_output,
        residual,
        output,
        injection_logits.stride(-2),
        block_output.stride(-2),
        residual.stride(-2),
        output.stride(-2),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_h=block_h,
        num_warps=4,
    )
    return output


def qwen4_hc_inject_combine_fake(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del injection_logits, block_output, hc_count
    return torch.empty_like(residual)


def can_use_hc_combine_norm_triton(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
) -> bool:
    """Return whether delayed combine+grouped-RMSNorm can use Triton.

    The injection view may retain the wider row stride of the packed
    down+inject projection.  The residual and block output are required to
    remain contiguous because the fused kernel writes a fresh contiguous
    multi-stream state and normalizes each branch in place conceptually.
    """

    return bool(
        can_use_hc_inject_triton(injection_logits, block_output, residual)
        and norm_weight.device == residual.device
        and norm_weight.dtype in (torch.bfloat16, torch.float16)
        and norm_weight.is_contiguous()
    )


@triton.jit
def _hc_combine_norm_kernel(
    block_ptr,
    residual_ptr,
    injection_ptr,
    weight_ptr,
    output_ptr,
    normed_ptr,
    stride_block_row,
    stride_residual_row,
    stride_injection_row,
    stride_output_row,
    stride_normed_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    weight_shared: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    """Fuse pending HC injection, materialization, and grouped RMSNorm.

    One program owns one token/branch pair.  All hidden-size tiles are kept
    in the program so the RMS reduction sees the same rounded-to-residual-dtype
    values that the unfused combine -> norm sequence would consume.
    """

    num_tiles: tl.constexpr = triton.cdiv(hidden_size, block_h)
    num_tiles_padded: tl.constexpr = triton.next_power_of_2(num_tiles)

    row = tl.program_id(0)
    branch = tl.program_id(1)
    tile_ids = tl.arange(0, num_tiles_padded)
    offsets = tile_ids[:, None] * block_h + tl.arange(0, block_h)[None, :]
    mask = offsets < hidden_size
    branch_offsets = branch * hidden_size + offsets
    weight_offsets = offsets if weight_shared else branch_offsets

    residual = tl.load(
        residual_ptr + row * stride_residual_row + branch_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    block = tl.load(
        block_ptr + row * stride_block_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    injection = tl.load(
        injection_ptr + row * stride_injection_row + branch,
    ).to(tl.float32)
    injection = 2.0 * tl.sigmoid(injection / hc_count)

    # Match the official fused kernel's explicit residual-dtype boundary.
    combined = (residual + block * injection).to(output_ptr.dtype.element_ty)
    tl.store(
        output_ptr + row * stride_output_row + branch_offsets,
        combined,
        mask=mask,
    )

    combined_f32 = combined.to(tl.float32)
    sum_sq = tl.sum(tl.sum(combined_f32 * combined_f32, axis=1), axis=0)
    inv_rms = tl.rsqrt(sum_sq / hidden_size + eps)
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    normalized = combined_f32 * inv_rms
    normalized += normalized * weight.to(tl.float32)
    tl.store(
        normed_ptr + row * stride_normed_row + branch_offsets,
        normalized,
        mask=mask,
    )


@triton.jit
def _hc_combine_norm_small_kernel(
    block_ptr,
    residual_ptr,
    injection_ptr,
    weight_ptr,
    output_ptr,
    normed_ptr,
    stride_block_row,
    stride_residual_row,
    stride_injection_row,
    stride_output_row,
    stride_normed_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    weight_shared: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
    inv_hidden: tl.constexpr,
    inv_hc: tl.constexpr,
) -> None:
    """Small-batch arm: (row, branch) programs, mulrec + num_warps=1 tuned."""

    num_tiles: tl.constexpr = triton.cdiv(hidden_size, block_h)
    num_tiles_padded: tl.constexpr = triton.next_power_of_2(num_tiles)

    row = tl.program_id(0)
    branch = tl.program_id(1)
    tile_ids = tl.arange(0, num_tiles_padded)
    offsets = tile_ids[:, None] * block_h + tl.arange(0, block_h)[None, :]
    mask = offsets < hidden_size
    branch_offsets = branch * hidden_size + offsets
    weight_offsets = offsets if weight_shared else branch_offsets

    residual = tl.load(
        residual_ptr + row * stride_residual_row + branch_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    block = tl.load(
        block_ptr + row * stride_block_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    injection = tl.load(
        injection_ptr + row * stride_injection_row + branch,
    ).to(tl.float32)
    injection = 2.0 * tl.sigmoid(injection * inv_hc)

    combined = (residual + block * injection).to(output_ptr.dtype.element_ty)
    tl.store(
        output_ptr + row * stride_output_row + branch_offsets,
        combined,
        mask=mask,
    )

    combined_f32 = combined.to(tl.float32)
    sum_sq = tl.sum(tl.sum(combined_f32 * combined_f32, axis=1), axis=0)
    inv_rms = tl.rsqrt(sum_sq * inv_hidden + eps)
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    normalized = combined_f32 * inv_rms
    normalized += normalized * weight.to(tl.float32)
    tl.store(
        normed_ptr + row * stride_normed_row + branch_offsets,
        normalized,
        mask=mask,
    )


@triton.jit
def _hc_combine_norm_row_kernel(
    block_ptr,
    residual_ptr,
    injection_ptr,
    weight_ptr,
    output_ptr,
    normed_ptr,
    stride_block_row,
    stride_residual_row,
    stride_injection_row,
    stride_output_row,
    stride_normed_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    weight_shared: tl.constexpr,
    eps: tl.constexpr,
    lanes: tl.constexpr,
    inv_hidden: tl.constexpr,
    inv_hc: tl.constexpr,
) -> None:
    """Whole-row arm: one [HC, LANES] tile per row.

    Block and weight loads are shared across streams, the injection sigmoid
    runs once per row as a [HC] vector, and a single axis=1 reduction tree
    replaces HC per-branch trees.  ``lanes`` must exceed ``hidden_size``
    (column-mask form): a [HC, LANES] tile with ``lanes == hidden_size``
    (exact multiple, no column mask) is a known silent-corruption trap on
    this backend, so the host dispatch guards ``lanes > hidden_size``.
    """

    row = tl.program_id(0)
    streams = tl.arange(0, hc_count)
    cols = tl.arange(0, lanes)
    col_mask = cols < hidden_size
    offs = streams[:, None] * hidden_size + cols[None, :]
    mask = col_mask[None, :]

    residual = tl.load(
        residual_ptr + row * stride_residual_row + offs, mask=mask, other=0.0
    ).to(tl.float32)
    block = tl.load(
        block_ptr + row * stride_block_row + cols, mask=col_mask, other=0.0
    ).to(tl.float32)
    injection = tl.load(injection_ptr + row * stride_injection_row + streams).to(
        tl.float32
    )
    injection_weight = 2.0 * tl.sigmoid(injection * inv_hc)

    # Match the official fused kernel's explicit residual-dtype boundary.
    combined = (residual + block[None, :] * injection_weight[:, None]).to(
        output_ptr.dtype.element_ty
    )
    tl.store(output_ptr + row * stride_output_row + offs, combined, mask=mask)

    combined_f32 = combined.to(tl.float32)
    sum_sq = tl.sum(combined_f32 * combined_f32, axis=1)
    inv_rms = tl.rsqrt(sum_sq * inv_hidden + eps)
    if weight_shared:
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    else:
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normalized = combined_f32 * inv_rms[:, None]
    normalized = normalized + normalized * weight
    tl.store(normed_ptr + row * stride_normed_row + offs, normalized, mask=mask)


@triton.jit
def _hc_combine_norm_batch_kernel(
    block_ptr,
    residual_ptr,
    injection_ptr,
    weight_ptr,
    output_ptr,
    normed_ptr,
    num_rows,
    stride_block_row,
    stride_residual_row,
    stride_injection_row,
    stride_output_row,
    stride_normed_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    weight_shared: tl.constexpr,
    eps: tl.constexpr,
    lanes: tl.constexpr,
    inv_hidden: tl.constexpr,
    inv_hc: tl.constexpr,
    rtile: tl.constexpr,
) -> None:
    """Row-batch arm: [RTILE, HC, LANES] tile per program.

    Keeps the grid within one wave for large batches.  All base addresses
    are scalar + per-dim arange arithmetic (no flattened 2D row bases).
    ``lanes`` must exceed ``hidden_size`` (column-mask form, see the
    whole-row arm docstring).
    """

    rows_idx = tl.program_id(0) * rtile + tl.arange(0, rtile)
    row_mask = rows_idx < num_rows
    streams = tl.arange(0, hc_count)
    cols = tl.arange(0, lanes)
    col_mask = cols < hidden_size

    offs = (
        rows_idx[:, None, None] * stride_residual_row
        + streams[None, :, None] * hidden_size
        + cols[None, None, :]
    )
    mask = row_mask[:, None, None] & col_mask[None, None, :]

    residual = tl.load(residual_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    block_offs = rows_idx[:, None] * stride_block_row + cols[None, :]
    block_mask = row_mask[:, None] & col_mask[None, :]
    block = tl.load(block_ptr + block_offs, mask=block_mask, other=0.0).to(tl.float32)
    injection_offs = rows_idx[:, None] * stride_injection_row + streams[None, :]
    injection = tl.load(
        injection_ptr + injection_offs, mask=row_mask[:, None], other=0.0
    ).to(tl.float32)
    injection_weight = 2.0 * tl.sigmoid(injection * inv_hc)

    combined = (residual + block[:, None, :] * injection_weight[:, :, None]).to(
        output_ptr.dtype.element_ty
    )
    tl.store(
        output_ptr
        + rows_idx[:, None, None] * stride_output_row
        + streams[None, :, None] * hidden_size
        + cols[None, None, :],
        combined,
        mask=mask,
    )

    combined_f32 = combined.to(tl.float32)
    sum_sq = tl.sum(combined_f32 * combined_f32, axis=2)
    inv_rms = tl.rsqrt(sum_sq * inv_hidden + eps)
    if weight_shared:
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        normalized = combined_f32 * inv_rms[:, :, None]
        normalized = normalized + normalized * weight[None, None, :]
    else:
        weight_offs = streams[None, :, None] * hidden_size + cols[None, None, :]
        weight = tl.load(
            weight_ptr + weight_offs,
            mask=col_mask[None, None, :],
            other=0.0,
        ).to(tl.float32)
        normalized = combined_f32 * inv_rms[:, :, None]
        normalized = normalized + normalized * weight
    tl.store(
        normed_ptr
        + rows_idx[:, None, None] * stride_normed_row
        + streams[None, :, None] * hidden_size
        + cols[None, None, :],
        normalized,
        mask=mask,
    )


def qwen4_hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused delayed HC combine + grouped Gemma RMSNorm."""

    if residual.ndim != 2 or block_output.ndim != 2 or injection_logits.ndim != 2:
        raise ValueError("Qwen4 HC combine_norm currently requires 2-D tensors")
    if hc_count <= 0 or residual.shape[-1] % hc_count:
        raise ValueError("Qwen4 HC combine_norm requires a valid HC count")
    hidden_size = residual.shape[-1] // hc_count
    if block_output.shape != (residual.shape[0], hidden_size):
        raise ValueError("Qwen4 HC combine_norm block output shape is invalid")
    if injection_logits.shape != (residual.shape[0], hc_count):
        raise ValueError("Qwen4 HC combine_norm injection shape is invalid")
    if norm_weight.numel() not in (hidden_size, residual.shape[-1]):
        raise ValueError("Qwen4 HC combine_norm weight shape is invalid")
    if not can_use_hc_combine_norm_triton(
        injection_logits, block_output, residual, norm_weight
    ):
        raise NotImplementedError(
            "Qwen4 HC combine_norm requires contiguous accelerator tensors"
        )

    if eps <= 0:
        raise ValueError("HC normalization epsilon must be positive")
    output = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    if not residual.numel():
        return output, normalized

    rows = residual.shape[0]
    weight_shared = norm_weight.numel() == hidden_size

    # GCU three-arm dispatch (Fixed Grid 48 SPCs, probed on Libra):
    #   rows*hc <= 48  -> (row, branch) programs, mulrec, num_warps=1
    #   rows <= 48     -> whole-row [HC, LANES] tile (shared loads, one tree)
    #   rows > 48      -> [RTILE, HC, LANES] row batch; RTILE/num_warps keyed
    #                     on lanes and rows (2026-11-11 rescan): narrow lanes
    #                     (<=1024) are tile-efficiency bound -> RTILE grows
    #                     with rows (fixed RTILE=2 was pathological from
    #                     rows~49 and 2-3x slower beyond 128); mid lanes
    #                     (<=2048) take RTILE=4 + nw=4; wide lanes (>=4096)
    #                     keep RTILE=2 for the DSM budget but need nw=4 for
    #                     load/store emission depth (L0 probe: reduction
    #                     tree is cheap, memory emission dominates).
    # Guards: hc_count power of two (tl.arange + exact 1/hc mulrec) and
    # lanes > hidden_size (column-mask form; exact-multiple [HC, LANES]
    # tiles are a known silent-corruption trap on this backend).  Shapes
    # outside the guards take the original generic kernel below.
    if hc_count > 0 and (hc_count & (hc_count - 1)) == 0:
        lanes = triton.next_power_of_2(hidden_size)
        if lanes > hidden_size:
            if rows * hc_count <= 48:
                _hc_combine_norm_small_kernel[(rows, hc_count)](
                    block_output,
                    residual,
                    injection_logits,
                    norm_weight,
                    output,
                    normalized,
                    block_output.stride(0),
                    residual.stride(0),
                    injection_logits.stride(0),
                    output.stride(0),
                    normalized.stride(0),
                    hidden_size=hidden_size,
                    hc_count=hc_count,
                    weight_shared=weight_shared,
                    eps=eps,
                    block_h=512 if hidden_size <= 1024 else 1024,
                    inv_hidden=1.0 / hidden_size,
                    inv_hc=1.0 / hc_count,
                    num_warps=1,
                )
                return output, normalized
            if rows <= 48:
                _hc_combine_norm_row_kernel[(rows,)](
                    block_output,
                    residual,
                    injection_logits,
                    norm_weight,
                    output,
                    normalized,
                    block_output.stride(0),
                    residual.stride(0),
                    injection_logits.stride(0),
                    output.stride(0),
                    normalized.stride(0),
                    hidden_size=hidden_size,
                    hc_count=hc_count,
                    weight_shared=weight_shared,
                    eps=eps,
                    lanes=lanes,
                    inv_hidden=1.0 / hidden_size,
                    inv_hc=1.0 / hc_count,
                    num_warps=1,
                )
                return output, normalized
            if lanes <= 1024:
                batch_tile = 4 if rows <= 192 else 8
                batch_warps = 2
            elif lanes <= 2048:
                batch_tile = 4
                batch_warps = 4
            else:
                batch_tile = 2
                batch_warps = 4
            _hc_combine_norm_batch_kernel[(triton.cdiv(rows, batch_tile),)](
                block_output,
                residual,
                injection_logits,
                norm_weight,
                output,
                normalized,
                rows,
                block_output.stride(0),
                residual.stride(0),
                injection_logits.stride(0),
                output.stride(0),
                normalized.stride(0),
                hidden_size=hidden_size,
                hc_count=hc_count,
                weight_shared=weight_shared,
                eps=eps,
                lanes=lanes,
                inv_hidden=1.0 / hidden_size,
                inv_hc=1.0 / hc_count,
                rtile=batch_tile,
                num_warps=batch_warps,
            )
            return output, normalized

    _hc_combine_norm_kernel[(rows, hc_count)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        output,
        normalized,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0),
        output.stride(0),
        normalized.stride(0),
        hidden_size=hidden_size,
        hc_count=hc_count,
        weight_shared=weight_shared,
        eps=eps,
        block_h=512,
        num_warps=8 if hidden_size > 2048 else 4,
    )
    return output, normalized


def _install() -> None:
    """Swap the GCU-adapted entry point into the generic module.

    Tests import ``from flaggems_vllm.ops.qwen4.hyperconnection import ...``
    which resolves against the already-imported generic module, so the
    SpecOpRegistrar top-level namespace swap alone cannot reach them
    (qsa/kpool backend-override precedent).  Only a pristine generic
    function is swapped, so explicit user or test patches always win.
    """
    import sys

    mod = sys.modules.get("flaggems_vllm.ops.qwen4.hyperconnection")
    if mod is None:
        return
    current = getattr(mod, "qwen4_hc_combine_norm", None)
    if current is not None and getattr(current, "__module__", None) == (
        "flaggems_vllm.ops.qwen4.hyperconnection"
    ):
        setattr(mod, "qwen4_hc_combine_norm", qwen4_hc_combine_norm)


try:
    _install()
except Exception:
    # pytest assertion-rewrite 上下文中 @triton.jit 源定位可能失败
    # (collect 期动态触碰 generic 模块触发);跳过接线不影响 op 路由
    # (SpecOpRegistrar 已保证),直连 import 入口的接线由运行期重试兜底。
    pass
__all__ = [
    "can_use_hc_triton",
    "can_use_hc_inject_triton",
    "can_use_hc_combine_norm_triton",
    "qwen4_grouped_gemma_rmsnorm",
    "qwen4_hc_gate_reduce",
    "qwen4_hc_inject_combine",
    "qwen4_hc_combine_norm",
]
