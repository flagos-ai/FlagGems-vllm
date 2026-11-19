# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 block-sparse GQA attention over a paged KV cache.

This module is a thin dispatch layer over FlagAttention's
``flag_attn.minimax_sparse_attention`` implementation, which owns the Triton
kernels (TMA/TLE prefill, kv-outer CSR reuse, split-K decode plus LSE merge).
Keeping a single implementation there avoids a second copy of the kernels
drifting out of sync with it.

Only the main-head attention consumer is exposed. The lightning indexer that
produces ``topk_idx`` is a separate operator and is not wrapped here.

Cache layout (vLLM packed pages, one sparse block == one 128-token KV page)::

    kv_cache:    [num_blocks, num_kv_heads, 128, 2 * head_dim]
                 K = [..., :head_dim]   V = [..., head_dim:]
    topk_idx:    [num_kv_heads, total_q, topk]   int32, -1 == hole
    block_table: [batch, max_blocks]             int32

``flag_attn`` is an optional dependency: it is imported lazily so that merely
importing ``flaggems_vllm`` never requires it, and an unsupported call raises
``NotImplementedError`` rather than falling back to a torch implementation.
"""

import functools

import torch

_FLAG_ATTN_HINT = (
    "MiniMax M3 sparse attention requires FlagAttention. Install it from "
    "https://github.com/flagos-ai/FlagAttention (it provides the "
    "flag_attn.minimax_sparse_attention Triton kernels)."
)

# One sparse block maps to exactly one KV page.
SPARSE_BLOCK_SIZE = 128

_SUPPORTED_KV_DTYPES = (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2)


@functools.lru_cache(maxsize=1)
def _backend():
    """Import the FlagAttention MSA entry points once, on first real call.

    Deferred so that importing this module (and therefore ``flaggems_vllm``)
    has no hard dependency on FlagAttention and triggers no CUDA access.
    """
    try:
        from flag_attn.minimax_sparse_attention import minimax_m3_sparse_attn as prefill
        from flag_attn.minimax_sparse_attention import (
            minimax_m3_sparse_attn_decode as decode,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NotImplementedError(_FLAG_ATTN_HINT) from exc
    return prefill, decode


def is_available() -> bool:
    """Whether the FlagAttention backend can be imported.

    Intended for callers and tests that need to probe support without
    provoking ``NotImplementedError``.
    """
    try:
        _backend()
    except NotImplementedError:
        return False
    return True


def _check_common(q, kv_cache, topk_idx, block_table, output, num_kv_heads, sm_scale):
    """Validate the contract shared by prefill and decode.

    Metadata reads only; contents of the device index tensors are never read
    back, so selected-ID and length bounds stay the caller's responsibility.
    """
    if not isinstance(num_kv_heads, int) or isinstance(num_kv_heads, bool):
        raise ValueError("num_kv_heads must be a host int")
    if num_kv_heads < 1:
        raise ValueError("num_kv_heads must be positive")
    if q.ndim != 3:
        raise ValueError("q must be [total_q, num_heads, head_dim]")
    # Dtype support is checked before the output/q agreement so that an
    # unsupported q dtype reports the unsupported dtype rather than a
    # mismatch against an output that happens to be BF16.
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("MiniMax M3 sparse attention requires BF16 Q/output")
    if output.shape != q.shape:
        raise ValueError("output must have the same shape as q")
    if output.dtype != q.dtype:
        raise ValueError("output must have the same dtype as q")
    num_heads = q.shape[1]
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if q.device.type != "cuda":
        raise NotImplementedError("MiniMax M3 sparse attention requires a CUDA device")
    if q.requires_grad or kv_cache.requires_grad:
        raise NotImplementedError("MiniMax M3 sparse attention is inference only")
    head_dim = q.shape[-1]
    if kv_cache.ndim != 4:
        raise ValueError(
            "kv_cache must be packed [num_blocks, num_kv_heads, 128, 2 * head_dim]"
        )
    if kv_cache.shape[1:] != (num_kv_heads, SPARSE_BLOCK_SIZE, 2 * head_dim):
        raise ValueError(
            "kv_cache must be [num_blocks, num_kv_heads, "
            f"{SPARSE_BLOCK_SIZE}, 2 * head_dim]; got {tuple(kv_cache.shape)}"
        )
    if kv_cache.dtype not in _SUPPORTED_KV_DTYPES:
        raise NotImplementedError(
            "kv_cache must be BF16 or FP8 (e4m3/e5m2); got " f"{kv_cache.dtype}"
        )
    if topk_idx.ndim != 3:
        raise ValueError("topk_idx must be [num_kv_heads, total_q, topk]")
    if topk_idx.shape[0] != num_kv_heads or topk_idx.shape[1] != q.shape[0]:
        raise ValueError(
            "topk_idx must be [num_kv_heads, total_q, topk]; got "
            f"{tuple(topk_idx.shape)}"
        )
    if block_table.ndim != 2:
        raise ValueError("block_table must be [batch, max_blocks]")
    for name, tensor in (("topk_idx", topk_idx), ("block_table", block_table)):
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32")
    for name, tensor in (
        ("kv_cache", kv_cache),
        ("topk_idx", topk_idx),
        ("block_table", block_table),
        ("output", output),
    ):
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
    if isinstance(sm_scale, bool) or not isinstance(sm_scale, (int, float)):
        raise ValueError("sm_scale must be a host float")
    if sm_scale != sm_scale or sm_scale in (float("inf"), float("-inf")):
        raise ValueError("sm_scale must be finite")


def _check_scales(q, k_scale, v_scale):
    if (k_scale is None) != (v_scale is None):
        raise ValueError("k_scale and v_scale must be provided together")
    if k_scale is None:
        return
    for name, tensor in (("k_scale", k_scale), ("v_scale", v_scale)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")


def minimax_m3_sparse_attn(
    q,
    kv_cache,
    topk_idx,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    num_kv_heads,
    sm_scale,
    output,
    k_scale=None,
    v_scale=None,
):
    """Block-sparse GQA prefill attention over the selected blocks.

    Writes into ``output`` and returns it. ``block_size_q == 1``: every query
    token carries its own selected-block list. FlagAttention picks the kernel
    (q-outer TMA/TLE, or the kv-outer CSR path on long BF16 prompts).

    Args:
        q: ``[total_q, num_heads, head_dim]`` BF16.
        kv_cache: ``[num_blocks, num_kv_heads, 128, 2 * head_dim]``.
        topk_idx: ``[num_kv_heads, total_q, topk]`` int32; ``-1`` is a hole.
        block_table: ``[batch, max_blocks]`` int32.
        cu_seqlens_q: ``[batch + 1]`` int32 cumulative query offsets.
        seq_lens: ``[batch]`` int32 total sequence lengths.
        prefix_lens: ``[batch]`` int32 cached prefix lengths.
        max_query_len: host int, max per-request query length.
        num_kv_heads: host int.
        sm_scale: host float softmax scale.
        output: ``[total_q, num_heads, head_dim]`` BF16, must not alias inputs.
        k_scale, v_scale: optional paired FP32 scale tensors for FP8 KV.

    Raises:
        NotImplementedError: unsupported dtype/layout/device, or FlagAttention
            is not installed. There is no torch compute fallback.
    """
    _check_common(q, kv_cache, topk_idx, block_table, output, num_kv_heads, sm_scale)
    _check_scales(q, k_scale, v_scale)
    if isinstance(max_query_len, bool) or not isinstance(max_query_len, int):
        raise ValueError("max_query_len must be a host int")
    if max_query_len < 0:
        raise ValueError("max_query_len must be non-negative")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q must be [batch + 1]")
    batch = cu_seqlens_q.numel() - 1
    for name, tensor in (("seq_lens", seq_lens), ("prefix_lens", prefix_lens)):
        if tensor.shape != (batch,):
            raise ValueError(f"{name} must be [batch] matching cu_seqlens_q")
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("seq_lens", seq_lens),
        ("prefix_lens", prefix_lens),
    ):
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
    if block_table.shape[0] != batch:
        raise ValueError("block_table batch dim must match cu_seqlens_q")
    if q.shape[0] == 0:
        return output
    if max_query_len == 0:
        raise ValueError("non-empty q requires a positive max_query_len")
    prefill, _ = _backend()
    prefill(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        output,
        k_scale,
        v_scale,
    )
    return output


def minimax_m3_sparse_attn_decode(
    q,
    kv_cache,
    topk_idx,
    block_table,
    seq_lens,
    num_kv_heads,
    sm_scale,
    output,
    decode_query_len,
    k_scale=None,
    v_scale=None,
):
    """Block-sparse GQA decode attention (split-K over the selected blocks).

    Writes into ``output`` and returns it. Decode parallelizes over the
    selected-block dimension instead of the query dimension, since one query
    token per request leaves the prefill kernels idle. FlagAttention chooses
    the chunk count and whether the LSE merge is needed.

    Args:
        q: ``[total_q, num_heads, head_dim]`` BF16, with
            ``total_q == num_reqs * decode_query_len``.
        kv_cache: ``[num_blocks, num_kv_heads, 128, 2 * head_dim]``.
        topk_idx: ``[num_kv_heads, total_q, topk]`` int32; ``-1`` is a hole.
        block_table: ``[num_reqs, max_blocks]`` int32.
        seq_lens: ``[num_reqs]`` int32.
        num_kv_heads: host int.
        sm_scale: host float softmax scale.
        output: ``[total_q, num_heads, head_dim]`` BF16, must not alias inputs.
        decode_query_len: host int, query tokens per request.
        k_scale, v_scale: optional paired FP32 scale tensors for FP8 KV.

    Raises:
        NotImplementedError: unsupported dtype/layout/device, or FlagAttention
            is not installed. There is no torch compute fallback.
    """
    _check_common(q, kv_cache, topk_idx, block_table, output, num_kv_heads, sm_scale)
    _check_scales(q, k_scale, v_scale)
    if isinstance(decode_query_len, bool) or not isinstance(decode_query_len, int):
        raise ValueError("decode_query_len must be a host int")
    if decode_query_len < 1:
        raise ValueError("decode_query_len must be positive")
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must be [num_reqs]")
    if seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be int32")
    if seq_lens.device != q.device:
        raise ValueError("seq_lens must be on the same device as q")
    if block_table.shape[0] != seq_lens.shape[0]:
        raise ValueError("block_table and seq_lens must agree on num_reqs")
    if q.shape[0] != seq_lens.shape[0] * decode_query_len:
        raise ValueError("total_q must equal num_reqs * decode_query_len")
    if q.shape[0] == 0:
        return output
    _, decode = _backend()
    decode(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        decode_query_len,
        k_scale,
        v_scale,
    )
    return output
