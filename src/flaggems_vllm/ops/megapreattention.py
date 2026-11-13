# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""PPU MegaPreAttention with SymmMem projection and Triton postprocessing."""

import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.mrope import _axis_for_pair
from flaggems_vllm.runtime import device as runtime_device
from flaggems_vllm.runtime import torch_device_fn


@triton.jit
def _megapreattention_post(
    projected,
    positions,
    q_norm,
    k_norm,
    rope,
    kv_cache,
    slots,
    query,
    key,
    tokens,
    eps,
    PROJ_STRIDE: tl.constexpr,
    POS_AXIS_STRIDE: tl.constexpr,
    POS_TOKEN_STRIDE: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    CACHE_BLOCK_STRIDE: tl.constexpr,
    CACHE_HEAD_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    CACHE_CAPACITY: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    token = (tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    head = tl.program_id(1)
    d = tl.arange(0, 256)
    valid = token < tokens
    is_query = head < Q_HEADS
    kv_head = head - Q_HEADS
    column = tl.where(is_query, head * 512, Q_HEADS * 512 + kv_head * 256)
    norm = q_norm if is_query else k_norm
    src = projected + token[:, None] * PROJ_STRIDE + column

    x = tl.load(src + d[None, :], valid[:, None], other=0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x, 1) / 256.0 + eps)
    w = tl.load(norm + d).to(tl.float32)
    normalized = (x * rms[:, None] * (1.0 + w[None, :])).to(tl.bfloat16)

    # NeoX pairs the two halves of the rotary prefix. Round both inputs to
    # BF16 before rotation, matching the separate Gemma RMSNorm output.
    partner_d = d ^ 32
    partner_x = tl.load(
        src + partner_d[None, :], valid[:, None] & (d[None, :] < 64), other=0
    ).to(tl.float32)
    partner_w = tl.load(norm + partner_d, d < 64, other=0).to(tl.float32)
    partner = (partner_x * rms[:, None] * (1.0 + partner_w[None, :])).to(tl.bfloat16)
    pair = d % 32
    axis = _axis_for_pair(pair, None, 0, SECTION_H, SECTION_W, True, False, 256)
    pos = tl.load(
        positions + token[:, None] * POS_TOKEN_STRIDE + axis[None, :] * POS_AXIS_STRIDE,
        valid[:, None] & (d[None, :] < 64),
        other=0,
    )
    phase = rope + pos * ROPE_STRIDE + pair[None, :]
    rotate = valid[:, None] & (d[None, :] < 64)
    cos = tl.load(phase, rotate, other=1).to(tl.float32)
    sin = tl.load(phase + 32, rotate, other=0).to(tl.float32)
    signed_sin = tl.where(d[None, :] < 32, -sin, sin)
    rotated = normalized.to(tl.float32) * cos + partner.to(tl.float32) * signed_sin
    result = tl.where(d[None, :] < 64, rotated.to(tl.bfloat16), normalized)

    if is_query:
        offset = (token[:, None] * Q_HEADS + head) * 256 + d[None, :]
        tl.store(query + offset, result, valid[:, None])
    else:
        offset = (token[:, None] * KV_HEADS + kv_head) * 256 + d[None, :]
        tl.store(key + offset, result, valid[:, None])
        slot = tl.load(slots + token * SLOT_STRIDE, valid, other=-1)
        cache_valid = valid & (slot >= 0) & (slot < CACHE_CAPACITY)
        cache_offset = (
            (slot // CACHE_BLOCK_SIZE) * CACHE_BLOCK_STRIDE
            + kv_head * CACHE_HEAD_STRIDE
            + (slot % CACHE_BLOCK_SIZE) * CACHE_TOKEN_STRIDE
        )
        target = kv_cache + cache_offset[:, None] + d[None, :]
        v = tl.load(src + KV_HEADS * 256 + d[None, :], cache_valid[:, None], other=0)
        tl.store(target, result, cache_valid[:, None])
        tl.store(target + 256, v, cache_valid[:, None])


def megapreattention(
    hidden,
    weight,
    positions,
    q_norm,
    k_norm,
    cos_sin_cache,
    kv_cache,
    slot_mapping,
    group,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    mrope_section,
    eps=1e-6,
):
    """Run BF16 TP2 pre-attention on T-Head PPU and update the KV cache in place.

    Each rank passes contiguous hidden shards [T/2, H] and weights
    [2*(local_q_heads + local_kv_heads)*256, H]. Weights pack Q and gate
    per query head, followed by all K heads and then all V heads. Head counts
    in the API are global. H is a multiple of 64 and at least 256, each shard
    has a positive multiple of 128 tokens, and head_dim/rotary_dim are 256/64.

    Positions are int64 [3,T] indexing the BF16 cos/sin table [P,64].
    The BF16 cache has non-overlapping layout [blocks, local_kv_heads,
    block_size, 512] with contiguous head elements and K followed by V.
    Int64 slots [T] must be unique and in bounds when nonnegative. Negative
    slots skip the cache write. Inputs must not alias the cache.

    Returns BF16 (query, key, value, gate). Q/K/V flatten their heads and
    the pre-activation gate has shape [T, local_q_heads, 256]. V and gate
    are strided views of the projection. Inference only. All ranks must
    call in the same order with matching metadata and serialize calls on one
    stream per group. Before capturing, warm up the largest planned hidden
    shard (in bytes) on every rank to reserve the group's SymmMem workspace,
    then warm up each shape to compile its kernels. Reset all dependent graphs
    before any call that can grow that workspace. This also applies to other
    SymmMem users sharing the process group.
    """
    if runtime_device.vendor_name != "thead" or hidden.device.type != "cuda":
        raise NotImplementedError("MegaPreAttention requires T-Head PPU")
    if group.size() != 2 or (head_dim, rotary_dim) != (256, 64):
        raise NotImplementedError("MegaPreAttention requires TP2, D256 and R64")
    if hidden.ndim != 2 or not hidden.is_contiguous():
        raise ValueError(
            "hidden must be a contiguous [local_tokens, hidden_size] tensor"
        )
    local_tokens, hidden_size = hidden.shape
    if not local_tokens or local_tokens % 128 or hidden_size < 256 or hidden_size % 64:
        raise NotImplementedError("unsupported token shard or hidden size")
    if (
        num_heads <= 0
        or num_heads % 2
        or num_kv_heads <= 0
        or (num_kv_heads > 1 and num_kv_heads % 2)
    ):
        raise ValueError("head counts must shard or replicate evenly across TP2")
    sections = tuple(mrope_section)
    if (
        len(sections) != 3
        or any(not isinstance(s, int) or s < 0 for s in sections)
        or sum(sections) != 32
        or not math.isfinite(eps)
        or eps < 0
    ):
        raise ValueError("invalid MRoPE sections or RMSNorm epsilon")

    device = hidden.device
    if any(
        x.dtype != torch.bfloat16 or x.device != device or x.requires_grad
        for x in (hidden, weight, q_norm, k_norm, cos_sin_cache, kv_cache)
    ):
        raise NotImplementedError(
            "MegaPreAttention requires BF16 inference tensors on one PPU"
        )
    tokens = local_tokens * 2
    q_heads, kv_heads = num_heads // 2, max(1, num_kv_heads // 2)
    q_width, kv_width = q_heads * head_dim, kv_heads * head_dim
    packed_width = 2 * (q_width + kv_width)
    if weight.shape != (packed_width, hidden_size) or not weight.is_contiguous():
        raise ValueError("invalid packed Q/gate/K/V weight")
    if any(w.shape != (head_dim,) or not w.is_contiguous() for w in (q_norm, k_norm)):
        raise ValueError("norm weights must be contiguous [256] tensors")
    if (
        positions.shape != (3, tokens)
        or slot_mapping.shape != (tokens,)
        or any(
            x.dtype != torch.int64 or x.device != device
            for x in (positions, slot_mapping)
        )
    ):
        raise ValueError(
            "expected int64 positions [3,T] and slots [T] on the input PPU"
        )
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[0] == 0
        or cos_sin_cache.shape[1] != rotary_dim
        or cos_sin_cache.stride(1) != 1
    ):
        raise ValueError("expected a cos/sin table [P,64] with a contiguous last axis")
    if (
        kv_cache.ndim != 4
        or kv_cache.shape[0] == 0
        or kv_cache.shape[2] == 0
        or kv_cache.shape[1] != kv_heads
        or kv_cache.shape[3] != 2 * head_dim
        or kv_cache.stride(3) != 1
    ):
        raise ValueError("invalid KV cache shape or last-axis stride")

    from torch.distributed import _symmetric_memory as symm_mem

    with torch_device_fn.device(device):
        symm_mem.enable_symm_mem_for_group(group.group_name)
        # SymmMem is the requested distributed projection, including GEMM.
        _, (projected,) = (
            torch.ops.symm_mem.fused_all_gather_matmul(  # allow-torch-compute
                hidden, [weight.t()], 0, group.group_name
            )
        )
        query = torch.empty((tokens, q_width), dtype=hidden.dtype, device=device)
        key = torch.empty((tokens, kv_width), dtype=hidden.dtype, device=device)
        # Fixed launch policy for the initial PPU baseline.
        block_t = 4
        _megapreattention_post[(triton.cdiv(tokens, block_t), q_heads + kv_heads)](
            projected,
            positions,
            q_norm,
            k_norm,
            cos_sin_cache,
            kv_cache,
            slot_mapping,
            query,
            key,
            tokens,
            eps,
            projected.stride(0),
            *positions.stride(),
            cos_sin_cache.stride(0),
            slot_mapping.stride(0),
            *kv_cache.stride()[:3],
            kv_cache.shape[2],
            kv_cache.shape[0] * kv_cache.shape[2],
            q_heads,
            kv_heads,
            sections[1],
            sections[2],
            block_t,
            num_warps=4,
            enable_fp_fusion=False,
        )
    value = projected[:, -kv_width:]
    gate = projected[:, : 2 * q_width].view(tokens, q_heads, 2 * head_dim)
    return query, key, value, gate[..., head_dim:]
