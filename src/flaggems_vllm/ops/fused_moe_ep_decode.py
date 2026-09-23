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

"""Opt-in Hopper expert-parallel decode path for fused MoE.

The path is deliberately strict and model-neutral: BF16, E288-to-local18,
top-8, H4096, I1280/I2048, clamp10, M=1..128 on NVIDIA H20. Callers must opt in
through ``enable_ep_decode_optimization`` in ``fused_experts_impl``. Nearby
shapes and every default call remain on the pre-existing implementation.

All GPU work is standard Triton. No-autotune exemption: the GEMM, compact
alignment, and reduction schedules are the fixed H20 winners measured in
FlagGems PR #5623 and are reachable only for the exact signature above.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.fused_moe_ep_m1 import _fused_moe_ep_m1_i2048_local_rank
from flaggems_vllm.ops.moe_sum import _moe_sum_ep

_GLOBAL_EXPERTS = 288
_LOCAL_EXPERTS = 18
_TOPK = 8
_HIDDEN_SIZE = 4096
_INTERMEDIATE_SIZES = (1280, 2048)
_MAX_TOKENS = 128
_CLAMP_LIMIT = 10.0

_GEMM1_BLOCK_M = 16
_GEMM1_BLOCK_N = 32
_GEMM1_BLOCK_K = 128
_GEMM2_BLOCK_M = 16
_GEMM2_BLOCK_N = 128
_GEMM2_BLOCK_K = 64


def _prepare_ep_decode_buffer(
    buffer: torch.Tensor | None,
    *,
    name: str,
    reference: torch.Tensor,
    required_numel: int,
) -> torch.Tensor:
    """Return the required flat slice of an optional caller-owned buffer."""
    if buffer is None:
        return torch.empty(
            required_numel,
            dtype=reference.dtype,
            device=reference.device,
        )
    if buffer.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {buffer.device}")
    if buffer.dtype != reference.dtype:
        raise ValueError(
            f"{name} must have dtype {reference.dtype}, got {buffer.dtype}"
        )
    if not buffer.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if buffer.numel() < required_numel:
        raise ValueError(
            f"{name} is too small: requires {required_numel} elements, "
            f"got {buffer.numel()}"
        )
    return buffer.view(-1)[:required_numel]


def _tensors_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return whether two contiguous tensor byte ranges overlap."""
    if lhs.numel() == 0 or rhs.numel() == 0 or lhs.device != rhs.device:
        return False
    lhs_begin = lhs.data_ptr()
    lhs_end = lhs_begin + lhs.numel() * lhs.element_size()
    rhs_begin = rhs.data_ptr()
    rhs_end = rhs_begin + rhs.numel() * rhs.element_size()
    return lhs_begin < rhs_end and rhs_begin < lhs_end


def _is_h20_device(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = device.index
    if torch.cuda.get_device_capability(index) != (9, 0):
        return False
    name = torch.cuda.get_device_name(index).replace(" ", "_")
    return "H20" in name.split("_")


def _select_ep_decode_plan(
    *,
    enabled: bool,
    is_h20: bool,
    num_tokens: int,
    local_num_experts: int,
    global_num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    clamp_limit: float | None,
    dtype: torch.dtype,
    has_expert_map: bool,
) -> str | None:
    """Pure host selector used by dispatch tests and the production gate."""
    if not (
        enabled is True
        and is_h20
        and 0 < num_tokens <= _MAX_TOKENS
        and local_num_experts == _LOCAL_EXPERTS
        and global_num_experts == _GLOBAL_EXPERTS
        and top_k == _TOPK
        and hidden_size == _HIDDEN_SIZE
        and intermediate_size in _INTERMEDIATE_SIZES
        and clamp_limit == _CLAMP_LIMIT
        and dtype == torch.bfloat16
        and has_expert_map
    ):
        return None
    if num_tokens == 1 and intermediate_size == 2048:
        return "adaptive_single_token"
    if num_tokens == 1:
        return "direct_single_token"
    return "compact_routed"


@triton.jit
def _ep_compact_count_prefix_init_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    expert_starts_ptr,
    expert_ranks_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    MAX_PADDED: tl.constexpr,
    INIT_BLOCK: tl.constexpr,
    MAX_BLOCKS_PER_EXPERT: tl.constexpr,
):
    route_offsets = tl.arange(0, BLOCK_ROUTES)
    route_mask = route_offsets < NUM_ROUTES
    global_experts_raw = tl.load(
        topk_ids_ptr + route_offsets,
        mask=route_mask,
        other=-1,
    )
    valid_global = (
        route_mask
        & (global_experts_raw >= 0)
        & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    )
    safe_global = tl.where(valid_global, global_experts_raw, 0).to(tl.int64)
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global,
        mask=valid_global,
        other=-1,
    )
    local_route = (
        valid_global
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )
    safe_local = tl.where(local_route, local_experts_raw, 0).to(tl.int32)
    counts = tl.histogram(safe_local, BLOCK_EXPERT, mask=local_route).to(tl.int32)

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < NUM_LOCAL_EXPERTS
    counts = tl.where(expert_mask, counts, 0)
    aligned_counts = tl.cdiv(counts, BLOCK_SIZE_M) * BLOCK_SIZE_M
    expert_starts = tl.cumsum(aligned_counts, axis=0) - aligned_counts
    total_tokens = tl.sum(aligned_counts, axis=0)

    tl.store(expert_starts_ptr + expert_offsets, expert_starts, mask=expert_mask)
    tl.store(expert_ranks_ptr + expert_offsets, 0, mask=expert_mask)
    tl.store(num_tokens_post_pad_ptr, total_tokens)

    init_offsets = tl.arange(0, INIT_BLOCK)
    for base in tl.static_range(0, MAX_PADDED, INIT_BLOCK):
        offsets = base + init_offsets
        tl.store(
            sorted_token_ids_ptr + offsets,
            NUM_ROUTES,
            mask=offsets < total_tokens,
        )

    for block_idx in tl.static_range(0, MAX_BLOCKS_PER_EXPERT):
        valid_block = expert_mask & (block_idx * BLOCK_SIZE_M < aligned_counts)
        output_block = expert_starts // BLOCK_SIZE_M + block_idx
        tl.store(
            expert_ids_ptr + output_block,
            expert_offsets,
            mask=valid_block,
        )


@triton.jit
def _ep_compact_scatter_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    expert_starts_ptr,
    expert_ranks_ptr,
    sorted_token_ids_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    route_offsets = tl.program_id(0) * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
    route_mask = route_offsets < NUM_ROUTES
    global_experts_raw = tl.load(
        topk_ids_ptr + route_offsets,
        mask=route_mask,
        other=-1,
    )
    valid_global = (
        route_mask
        & (global_experts_raw >= 0)
        & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    )
    safe_global = tl.where(valid_global, global_experts_raw, 0).to(tl.int64)
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global,
        mask=valid_global,
        other=-1,
    )
    local_route = (
        valid_global
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )
    safe_local = tl.where(local_route, local_experts_raw, 0).to(tl.int32)
    ranks = tl.atomic_add(expert_ranks_ptr + safe_local, 1, mask=local_route)
    starts = tl.load(expert_starts_ptr + safe_local, mask=local_route, other=0)
    tl.store(
        sorted_token_ids_ptr + starts + ranks,
        route_offsets,
        mask=local_route,
    )


def _align_ep_routes(
    topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_routes = topk_ids.numel()
    max_num_tokens_padded = num_routes + _LOCAL_EXPERTS * (_GEMM1_BLOCK_M - 1)
    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (triton.cdiv(max_num_tokens_padded, _GEMM1_BLOCK_M),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    expert_starts = torch.empty(
        (_LOCAL_EXPERTS,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ranks = torch.empty_like(expert_starts)
    return (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_starts,
        expert_ranks,
    )


def _launch_ep_alignment(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_ranks: torch.Tensor,
) -> None:
    num_routes = topk_ids.numel()
    max_num_tokens_padded = sorted_token_ids.numel()
    _ep_compact_count_prefix_init_kernel[(1,)](
        topk_ids,
        expert_map,
        expert_starts,
        expert_ranks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        NUM_ROUTES=num_routes,
        NUM_GLOBAL_EXPERTS=_GLOBAL_EXPERTS,
        NUM_LOCAL_EXPERTS=_LOCAL_EXPERTS,
        BLOCK_SIZE_M=_GEMM1_BLOCK_M,
        BLOCK_EXPERT=triton.next_power_of_2(_LOCAL_EXPERTS),
        BLOCK_ROUTES=triton.next_power_of_2(num_routes),
        MAX_PADDED=max_num_tokens_padded,
        INIT_BLOCK=256,
        MAX_BLOCKS_PER_EXPERT=triton.cdiv(num_routes, _GEMM1_BLOCK_M),
        num_warps=4,
    )
    scatter_block = 128
    _ep_compact_scatter_kernel[(triton.cdiv(num_routes, scatter_block),)](
        topk_ids,
        expert_map,
        expert_starts,
        expert_ranks,
        sorted_token_ids,
        NUM_ROUTES=num_routes,
        NUM_GLOBAL_EXPERTS=_GLOBAL_EXPERTS,
        NUM_LOCAL_EXPERTS=_LOCAL_EXPERTS,
        BLOCK_ROUTES=scatter_block,
        num_warps=4,
    )


@triton.jit
def _ep_decode_gemm1_kernel(
    hidden_ptr,
    w1_ptr,
    cache2_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    stride_hidden_m,
    stride_hidden_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_cache_route,
    stride_cache_n,
    NUM_ROUTES: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    DIRECT_ROUTE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(INTERMEDIATE_SIZE, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    if DIRECT_ROUTE:
        offs_token = tl.where(offs_m == 0, pid_m, NUM_ROUTES).to(tl.int64)
        global_expert_raw = tl.load(topk_ids_ptr + pid_m)
        valid_global = (global_expert_raw >= 0) & (global_expert_raw < 288)
        safe_global = tl.where(valid_global, global_expert_raw, 0).to(tl.int64)
        local_expert_raw = tl.load(
            expert_map_ptr + safe_global,
            mask=valid_global,
            other=-1,
        )
        valid_local = valid_global & (local_expert_raw >= 0) & (local_expert_raw < 18)
        if not valid_local:
            return
        local_expert = local_expert_raw.to(tl.int64)
    else:
        num_tokens_post_pad = tl.load(num_tokens_post_pad_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_pad:
            return
        token_offsets = pid_m * BLOCK_SIZE_M + offs_m
        offs_token = tl.load(sorted_token_ids_ptr + token_offsets).to(tl.int64)
        local_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    token_mask = offs_token < NUM_ROUTES
    token_ids = offs_token // 8
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    pair_offsets = tl.arange(0, BLOCK_SIZE_N * 2).to(tl.int64)
    tile_start = pid_n * BLOCK_SIZE_N
    pair_columns = tl.where(
        pair_offsets < BLOCK_SIZE_N,
        tile_start + pair_offsets,
        INTERMEDIATE_SIZE + tile_start + pair_offsets - BLOCK_SIZE_N,
    )
    output_columns = tile_start + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)

    a_ptrs = (
        hidden_ptr
        + token_ids[:, None] * stride_hidden_m
        + offs_k[None, :] * stride_hidden_k
    )
    b_ptrs = (
        w1_ptr
        + local_expert * stride_w1_e
        + pair_columns[None, :] * stride_w1_n
        + offs_k[:, None] * stride_w1_k
    )
    pair_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
    for _ in range(0, 4096, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        pair_acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_hidden_k
        b_ptrs += BLOCK_SIZE_K * stride_w1_k

    gate_up = tl.trans(
        tl.reshape(pair_acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N)),
        (0, 2, 1),
    )
    gate_acc, up_acc = tl.split(gate_up)
    # Preserve the prior GEMM1 BF16 materialization before clamp and SiLU.
    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    up = up_acc.to(tl.bfloat16).to(tl.float32)
    gate = tl.minimum(gate, 10.0)
    up = tl.minimum(tl.maximum(up, -10.0), 10.0)
    activated = tl.fdiv(gate, 1.0 + tl.exp(-gate)) * up
    cache_ptrs = (
        cache2_ptr
        + offs_token[:, None] * stride_cache_route
        + output_columns[None, :] * stride_cache_n
    )
    tl.store(cache_ptrs, activated, mask=token_mask[:, None])


@triton.jit
def _ep_decode_gemm2_kernel(
    cache2_ptr,
    w2_ptr,
    cache3_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    stride_cache2_route,
    stride_cache2_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_cache3_route,
    stride_cache3_n,
    stride_weight_route,
    NUM_ROUTES: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    DIRECT_ROUTE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(4096, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    if DIRECT_ROUTE:
        offs_token = tl.where(offs_m == 0, pid_m, NUM_ROUTES).to(tl.int64)
        global_expert_raw = tl.load(topk_ids_ptr + pid_m)
        valid_global = (global_expert_raw >= 0) & (global_expert_raw < 288)
        safe_global = tl.where(valid_global, global_expert_raw, 0).to(tl.int64)
        local_expert_raw = tl.load(
            expert_map_ptr + safe_global,
            mask=valid_global,
            other=-1,
        )
        valid_local = valid_global & (local_expert_raw >= 0) & (local_expert_raw < 18)
        if not valid_local:
            return
        local_expert = local_expert_raw.to(tl.int64)
    else:
        num_tokens_post_pad = tl.load(num_tokens_post_pad_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_pad:
            return
        token_offsets = pid_m * BLOCK_SIZE_M + offs_m
        offs_token = tl.load(sorted_token_ids_ptr + token_offsets).to(tl.int64)
        local_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    token_mask = offs_token < NUM_ROUTES
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        cache2_ptr
        + offs_token[:, None] * stride_cache2_route
        + offs_k[None, :] * stride_cache2_k
    )
    b_ptrs = (
        w2_ptr
        + local_expert * stride_w2_e
        + offs_n[None, :] * stride_w2_n
        + offs_k[:, None] * stride_w2_k
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, INTERMEDIATE_SIZE, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_cache2_k
        b_ptrs += BLOCK_SIZE_K * stride_w2_k

    route_weight = tl.load(
        topk_weights_ptr + offs_token * stride_weight_route,
        mask=token_mask,
        other=0.0,
    ).to(tl.float32)
    weighted = accumulator * route_weight[:, None]
    cache_ptrs = (
        cache3_ptr
        + offs_token[:, None] * stride_cache3_route
        + offs_n[None, :] * stride_cache3_n
    )
    tl.store(cache_ptrs, weighted, mask=token_mask[:, None])


def _validate_inputs(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
    apply_router_weight_on_input: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    ocp_mx_scheme: str | None,
    per_channel_quant: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    block_shape: Optional[list[int]],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    gemm1_clamp_limit: float | None,
) -> str:
    if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise NotImplementedError("optimized EP decode requires rank-2/3 inputs")
    input_tensors = (hidden_states, w1, w2, topk_weights, topk_ids)
    if expert_map is not None:
        input_tensors = (*input_tensors, expert_map)
    if any(tensor.requires_grad for tensor in input_tensors):
        raise NotImplementedError("optimized EP decode is inference-only")
    num_tokens, hidden_size = hidden_states.shape
    local_num_experts, gate_up_size, w1_hidden_size = w1.shape
    w2_experts, w2_hidden_size, intermediate_size = w2.shape
    top_k = topk_ids.shape[1] if topk_ids.ndim == 2 else -1
    plan = _select_ep_decode_plan(
        enabled=True,
        is_h20=_is_h20_device(hidden_states.device),
        num_tokens=num_tokens,
        local_num_experts=local_num_experts,
        global_num_experts=global_num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        clamp_limit=gemm1_clamp_limit,
        dtype=hidden_states.dtype,
        has_expert_map=expert_map is not None,
    )
    if plan is None:
        raise NotImplementedError(
            "enable_ep_decode_optimization requires H20 BF16, M=1..128, "
            "E288-to-local18, topk=8, H=4096, I=1280/2048, clamp=10"
        )

    if (
        gate_up_size != 2 * intermediate_size
        or w1_hidden_size != hidden_size
        or w2_experts != local_num_experts
        or w2_hidden_size != hidden_size
        or topk_ids.shape != (num_tokens, _TOPK)
        or topk_weights.shape != topk_ids.shape
    ):
        raise NotImplementedError("optimized EP decode tensor shapes do not match")
    if expert_map is None or expert_map.shape != (_GLOBAL_EXPERTS,):
        raise NotImplementedError("optimized EP decode requires a 288-entry expert_map")
    if not (
        w1.dtype == w2.dtype == torch.bfloat16
        and topk_weights.dtype in (torch.bfloat16, torch.float32)
        and topk_ids.dtype in (torch.int32, torch.int64)
        and expert_map.dtype in (torch.int32, torch.int64)
    ):
        raise NotImplementedError(
            "optimized EP decode dtype combination is unsupported"
        )
    tensors = (hidden_states, w1, w2, topk_weights, topk_ids, expert_map)
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("optimized EP decode tensors must share one device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise NotImplementedError("optimized EP decode requires contiguous tensors")
    if activation != "silu" or apply_router_weight_on_input:
        raise NotImplementedError(
            "optimized EP decode supports bias-free post-activation router weights"
        )
    if (
        any(
            (
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                per_channel_quant,
            )
        )
        or ocp_mx_scheme is not None
    ):
        raise NotImplementedError("optimized EP decode supports unquantized BF16 only")
    if any(
        value is not None
        for value in (
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            w1_bias,
            w2_bias,
        )
    ):
        raise NotImplementedError(
            "optimized EP decode does not support scales, zero-points, block "
            "quantization, or GEMM biases"
        )
    return plan


def _fused_moe_ep_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    inplace: bool,
    activation: str,
    apply_router_weight_on_input: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    ocp_mx_scheme: str | None,
    per_channel_quant: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    block_shape: Optional[list[int]],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    gemm1_clamp_limit: float | None,
    output: Optional[torch.Tensor],
    intermediate_cache13: Optional[torch.Tensor],
    intermediate_cache2: Optional[torch.Tensor],
) -> torch.Tensor:
    """Execute the explicitly enabled, exact-signature EP decode path.

    Caller-owned buffers are optional. ``intermediate_cache13`` backs the
    GEMM2 route rows; ``intermediate_cache2`` backs the fused GEMM1 activation.
    The output may alias cache2 because GEMM2 consumes all activation rows
    before the EP combine writes the final result.
    """
    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")
    if output is not None:
        if output.shape != hidden_states.shape:
            raise ValueError(
                f"output must have shape {tuple(hidden_states.shape)}, "
                f"got {tuple(output.shape)}"
            )
        if output.device != hidden_states.device:
            raise ValueError(
                f"output must be on {hidden_states.device}, got {output.device}"
            )
        if output.dtype != hidden_states.dtype:
            raise ValueError(
                f"output must have dtype {hidden_states.dtype}, got {output.dtype}"
            )
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")

    plan = _validate_inputs(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
    assert expert_map is not None

    num_tokens = hidden_states.shape[0]
    intermediate_size = w2.shape[2]
    num_routes = num_tokens * _TOPK
    cache2 = _prepare_ep_decode_buffer(
        intermediate_cache2,
        name="intermediate_cache2",
        reference=hidden_states,
        required_numel=num_routes * intermediate_size,
    ).view(num_routes, intermediate_size)
    cache3 = _prepare_ep_decode_buffer(
        intermediate_cache13,
        name="intermediate_cache13",
        reference=hidden_states,
        required_numel=num_routes * _HIDDEN_SIZE,
    ).view(num_routes, _HIDDEN_SIZE)
    out = hidden_states if inplace else output
    if out is None:
        out = torch.empty_like(hidden_states)

    protected_inputs = {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "expert_map": expert_map,
    }
    for cache_name, cache in (
        ("intermediate_cache13", cache3),
        ("intermediate_cache2", cache2),
    ):
        for input_name, input_tensor in protected_inputs.items():
            if _tensors_overlap(cache, input_tensor):
                raise ValueError(f"{cache_name} must not overlap {input_name}")
    if _tensors_overlap(cache3, cache2):
        raise ValueError(
            "intermediate_cache13 and intermediate_cache2 must not overlap"
        )
    if not inplace and _tensors_overlap(out, hidden_states):
        raise ValueError("output must not overlap hidden_states; use inplace=True")
    for input_name, input_tensor in protected_inputs.items():
        if input_name == "hidden_states":
            continue
        if _tensors_overlap(out, input_tensor):
            raise ValueError(f"output must not overlap {input_name}")
    if _tensors_overlap(out, cache3):
        raise ValueError("output must not overlap intermediate_cache13")

    if plan == "adaptive_single_token":
        return _fused_moe_ep_m1_i2048_local_rank(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            expert_map,
            cache2,
            cache3,
            out,
            use_singleton_gemv=True,
        )

    direct_route = plan == "direct_single_token"
    if direct_route:
        # These pointers are compile-time unused in direct mode. Reusing input
        # pointers avoids allocations or scalar fill kernels.
        sorted_token_ids = topk_ids
        expert_ids = topk_ids
        num_tokens_post_pad = topk_ids
        padded_route_capacity = num_routes * _GEMM1_BLOCK_M
    else:
        (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_starts,
            expert_ranks,
        ) = _align_ep_routes(topk_ids)
        _launch_ep_alignment(
            topk_ids,
            expert_map,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_starts,
            expert_ranks,
        )
        padded_route_capacity = sorted_token_ids.numel()

    gemm1_grid = (
        triton.cdiv(padded_route_capacity, _GEMM1_BLOCK_M)
        * triton.cdiv(intermediate_size, _GEMM1_BLOCK_N),
    )
    _ep_decode_gemm1_kernel[gemm1_grid](
        hidden_states,
        w1,
        cache2,
        topk_ids,
        expert_map,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        cache2.stride(0),
        cache2.stride(1),
        NUM_ROUTES=num_routes,
        INTERMEDIATE_SIZE=intermediate_size,
        DIRECT_ROUTE=direct_route,
        BLOCK_SIZE_M=_GEMM1_BLOCK_M,
        BLOCK_SIZE_N=_GEMM1_BLOCK_N,
        BLOCK_SIZE_K=_GEMM1_BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    gemm2_grid = (
        triton.cdiv(padded_route_capacity, _GEMM2_BLOCK_M)
        * triton.cdiv(_HIDDEN_SIZE, _GEMM2_BLOCK_N),
    )
    _ep_decode_gemm2_kernel[gemm2_grid](
        cache2,
        w2,
        cache3,
        topk_weights,
        topk_ids,
        expert_map,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cache2.stride(0),
        cache2.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        cache3.stride(0),
        cache3.stride(1),
        topk_weights.stride(1),
        NUM_ROUTES=num_routes,
        INTERMEDIATE_SIZE=intermediate_size,
        DIRECT_ROUTE=direct_route,
        BLOCK_SIZE_M=_GEMM2_BLOCK_M,
        BLOCK_SIZE_N=_GEMM2_BLOCK_N,
        BLOCK_SIZE_K=_GEMM2_BLOCK_K,
        num_warps=4,
        num_stages=4,
    )
    _moe_sum_ep(
        cache3.view(num_tokens, _TOPK, _HIDDEN_SIZE),
        out,
        topk_ids,
        expert_map,
        _LOCAL_EXPERTS,
    )
    return out


__all__ = []
