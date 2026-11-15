# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""PPU MegaPreAttention with native sailSHMEM and tile-level CTA scheduling."""

import math

import torch

from flaggems_vllm.runtime import device as runtime_device
from flaggems_vllm.runtime import torch_device_fn


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
    """Run BF16 pre-attention on single-node PPU TP and update KV in place.

    Each rank passes contiguous hidden shards [T/TP, H] and weights
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
    the pre-activation gate has shape [T, local_q_heads, 256].

    All ranks must call in the same order with matching metadata. Serialize
    launches and graph replay on each rank. Warm up every shape and layout
    before capture. Workspace addresses remain valid across shapes until
    clear_megapreattention_cache is called collectively after resetting graphs.
    """
    if runtime_device.vendor_name != "thead" or hidden.device.type != "cuda":
        raise NotImplementedError("MegaPreAttention requires T-Head PPU")
    tp = group.size()
    if (head_dim, rotary_dim) != (256, 64):
        raise NotImplementedError("MegaPreAttention requires D256 and R64")
    if hidden.ndim != 2 or not hidden.is_contiguous():
        raise ValueError(
            "hidden must be a contiguous [local_tokens, hidden_size] tensor"
        )
    local_tokens, hidden_size = hidden.shape
    if not local_tokens or local_tokens % 128 or hidden_size < 256 or hidden_size % 64:
        raise NotImplementedError("unsupported token shard or hidden size")
    if (
        num_heads <= 0
        or num_heads % tp
        or num_kv_heads <= 0
        or (num_kv_heads % tp and tp % num_kv_heads)
    ):
        raise ValueError("head counts must shard or replicate evenly across TP ranks")
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
    tokens = local_tokens * tp
    q_heads, kv_heads = num_heads // tp, max(1, num_kv_heads // tp)
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

    from ._megapreattention_kernel import megapreattention_kernel
    from ._sailshmem import runtime_for

    block_m, block_k, group_m, stages = 64, 64, 4, 3
    comm_m = 256 if local_tokens % 256 == 0 else 128
    if local_tokens >= 4096 and local_tokens % 512 == 0:
        comm_m = 512
    local_tiles = local_tokens // block_m
    comm_tiles = local_tokens // comm_m
    # Keep most of the device available for projection as messages accumulate.
    comm_jobs = min(comm_tiles, 4) if tp > 1 else 0
    grid = (comm_jobs + tp * local_tiles * 2 * (q_heads + kv_heads),)
    with torch_device_fn.device(device):
        runtime = runtime_for(group, device)
        workspace_bytes = (
            tokens * hidden_size * hidden.element_size() + tp * (comm_tiles + 1) * 8
        )
        workspace_id = (tp, local_tokens, hidden_size, comm_m, grid[0])
        symmetric, tickets = runtime.workspace(workspace_id, workspace_bytes)
        query = hidden.new_empty((tokens, q_width))
        key = hidden.new_empty((tokens, kv_width))
        value = hidden.new_empty((tokens, kv_width))
        gate = hidden.new_empty((tokens, q_heads, head_dim))
        args = (
            hidden,
            weight,
            positions,
            q_norm,
            k_norm,
            cos_sin_cache,
            kv_cache,
            slot_mapping,
            query,
            key,
            value,
            gate,
            tickets,
            symmetric,
            runtime.rank,
            eps,
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
            tp,
            local_tiles,
            hidden_size,
            comm_m,
            comm_jobs,
            block_m,
            block_k,
            group_m,
            stages,
        )
        options = dict(
            num_warps=8,
            enable_fp_fusion=False,
            extern_libs={"sailshmem": runtime.bitcode},
        )
        kernel = megapreattention_kernel.warmup(*args, grid=grid, **options)
        if tp > 1:
            runtime.register_kernel(kernel)
        megapreattention_kernel[grid](*args, **options)
    return query, key, value, gate


def clear_megapreattention_cache():
    """Collectively free native workspace after all dependent graphs are reset."""
    from ._sailshmem import clear_workspace

    clear_workspace()
