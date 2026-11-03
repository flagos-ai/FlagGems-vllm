# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Noncausal ViT attention through FlagGems, without vLLM extensions.

Import this module before tracing/capture so its custom ops and fake kernels
are registered. Sequence boundaries remain device inputs on graph replay.
"""

import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.attention import flash_attn_varlen_func
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def _pack_kernel(
    X,
    Y,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SB: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    SD: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = i // (H * D)
    offset = token // S * SB + token % S * ST + (i // D % H) * SH + i % D * SD
    if MODE == 0:
        value = tl.full((BLOCK,), 0.0, tl.float32)
    else:
        value = tl.load(X + offset, mask=i < N, other=0)
        if MODE == -1:
            value = -value
    tl.store(Y + i, value, mask=i < N)


def _pack(x, mode=1):
    tokens = math.prod(x.shape[:-2])
    output_shape = (tokens, x.shape[-2], x.shape[-1])
    if x.is_contiguous() and mode == 1:
        return x.view(output_shape)
    y = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if tokens:
        if x.ndim == 3:
            sequence, sb, st, sh, sd = x.shape[0], 0, *x.stride()
        else:
            sequence = x.shape[1]
            sb, st, sh, sd = x.stride()
        with torch_device_fn.device(x.device):
            _pack_kernel[(triton.cdiv(y.numel(), 1024),)](
                x,
                y,
                y.numel(),
                sequence,
                x.shape[-2],
                x.shape[-1],
                sb,
                st,
                sh,
                sd,
                mode,
                BLOCK=1024,
            )
    return y


@libentry()
@triton.jit
def _uniform_cu(
    Q, K, B: tl.constexpr, SQ: tl.constexpr, SK: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Q + i, i * SQ, mask=i <= B)
    tl.store(K + i, i * SK, mask=i <= B)


def supports_vit_attention(head_size, dtype):
    """Pure metadata gate for the dimensions validated by this adapter."""
    return head_size in (64, 80) and dtype in (torch.float16, torch.bfloat16)


def _check_qkv(q, k, v, dimensions):
    if q.device.type != "cuda":
        raise NotImplementedError("ViT attention requires a CUDA device")
    if q.ndim != dimensions or k.ndim != dimensions or v.ndim != dimensions:
        raise ValueError(f"Q/K/V must all have {dimensions} dimensions")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("ViT FlagGems attention requires FP16 or BF16 Q/K/V")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q/K/V dtypes must match")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V devices must match")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("K/V shapes and Q/K head dimensions must match")
    if not supports_vit_attention(q.shape[-1], q.dtype):
        raise ValueError("ViT adapter supports validated head dimensions 64/80")
    if k.shape[-2] <= 0 or q.shape[-2] % k.shape[-2]:
        raise ValueError("KV heads must divide Q heads")


def _check_cu(cu_q, cu_k, device):
    if cu_q.ndim != 1 or cu_k.ndim != 1 or cu_q.shape != cu_k.shape:
        raise ValueError("Q/K cumulative lengths must be equal-sized vectors")
    if cu_q.numel() < 2:
        raise ValueError("Cumulative lengths must include a start and end")
    if cu_q.dtype != torch.int32 or cu_k.dtype != torch.int32:
        raise ValueError("Cumulative lengths must be int32")
    if cu_q.device != device or cu_k.device != device:
        raise ValueError("Cumulative lengths must be on the Q/K/V device")
    if not cu_q.is_contiguous() or not cu_k.is_contiguous():
        raise ValueError("Cumulative lengths must be contiguous")


def _varlen_impl(q, k, v, cu_q, cu_k, scale):
    _check_qkv(q, k, v, 3)
    _check_cu(cu_q, cu_k, q.device)
    if scale is not None and not math.isfinite(scale):
        raise ValueError("Attention scale must be finite")
    if q.shape[0] == 0:
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if k.shape[0] == 0:
        return _pack(q, mode=0)

    # The existing kernel requires identical K/V strides. In M3, K is a
    # rotated contiguous buffer but V is a slice of the interleaved QKV buffer.
    # These copies also preserve the flattening order for other strided inputs.
    k, v = _pack(k), _pack(v)
    # FlagGems applies its border mask before multiplying by scale. A zero
    # scale would turn masked -inf into NaN, and negative scales reverse the
    # border mask. Transform Q exactly so the existing kernel sees a positive
    # scale while preserving the requested finite-input attention semantics.
    if scale == 0.0:
        q = _pack(q, mode=0)
        scale = 1.0
    elif scale is not None and scale < 0.0:
        q = _pack(q, mode=-1)
        scale = -scale
    else:
        q = _pack(q)
    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        # Safe shape-derived upper bounds; no .item() or host copy of metadata.
        # They remain valid when sequence boundaries change on graph replay.
        max_seqlen_q=q.shape[0],
        max_seqlen_k=k.shape[0],
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
        fa_version=2,
    )


@torch.library.custom_op("flaggems_vllm::vision_flash_attn_varlen", mutates_args=())
def _vision_flash_attn_varlen_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    return _varlen_impl(q, k, v, cu_seqlens_q, cu_seqlens_k, scale)


@_vision_flash_attn_varlen_op.register_fake
def _vision_flash_attn_varlen_fake(q, k, v, cu_seqlens_q, cu_seqlens_k, scale):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


def vision_flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, scale=None):
    """Packed [tokens, heads, dim] attention with distinct Q/K boundaries."""
    return _vision_flash_attn_varlen_op(q, k, v, cu_seqlens_q, cu_seqlens_k, scale)


@torch.library.custom_op("flaggems_vllm::vit_flash_attn_wrapper", mutates_args=())
def _vit_flash_attn_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: torch.Tensor | None,
) -> torch.Tensor:
    _check_qkv(q, k, v, 4)
    if q.shape[0] != batch_size or k.shape[0] != batch_size:
        raise ValueError("Q/K batch dimensions must equal batch_size")
    if max_seqlen is not None and max_seqlen.numel() != 1:
        raise ValueError("max_seqlen must be a scalar tensor")

    total_q = q.shape[0] * q.shape[1]
    total_k = k.shape[0] * k.shape[1]
    if total_q == 0:
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if cu_seqlens is None:
        cu_q = torch.empty((batch_size + 1,), device=q.device, dtype=torch.int32)
        cu_k = torch.empty_like(cu_q)
        with torch_device_fn.device(q.device):
            _uniform_cu[(triton.cdiv(batch_size + 1, 256),)](
                cu_q,
                cu_k,
                batch_size,
                q.shape[1],
                k.shape[1],
                BLOCK=256,
            )
    else:
        if total_q != total_k:
            raise ValueError(
                "Shared cu_seqlens requires equal total Q/K tokens; use "
                "vision_flash_attn_varlen for packed cross attention"
            )
        cu_q = cu_k = cu_seqlens

    q_flat, k_flat, v_flat = _pack(q), _pack(k), _pack(v)
    output = _varlen_impl(q_flat, k_flat, v_flat, cu_q, cu_k, scale)
    # is_rocm_aiter/fa_version select upstream compiled libraries. This
    # callable explicitly selects the existing FlagGems Triton implementation.
    return output.view(q.shape)


@_vit_flash_attn_op.register_fake
def _vit_flash_attn_fake(
    q, k, v, batch_size, is_rocm_aiter, fa_version, scale, cu_seqlens, max_seqlen
):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


def vit_flash_attn_wrapper(
    q,
    k,
    v,
    batch_size,
    is_rocm_aiter=False,
    fa_version=None,
    scale=None,
    cu_seqlens=None,
    max_seqlen=None,
):
    """vLLM-compatible [batch, tokens, heads, dim] ViT attention callable.

    The optional shared cumulative lengths describe packed self attention.
    Without them, each batch row is independent, including different Q/K
    sequence lengths. max_seqlen is accepted for API compatibility; the launch
    uses safe shape upper bounds, so it never reads a tensor scalar on the host.
    """
    return _vit_flash_attn_op(
        q,
        k,
        v,
        batch_size,
        is_rocm_aiter,
        fa_version,
        scale,
        cu_seqlens,
        max_seqlen,
    )
