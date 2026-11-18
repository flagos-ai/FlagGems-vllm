# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The vLLM team.
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
# Frozen vLLM gemma4.py at 5d83998e927f3081c804932e6586bd7312a424e8.
# Reference only: never imported by production operators.
import torch
import triton
import triton.language as tl


@triton.jit
def _gemma4_routing_kernel(
    gating_ptr,
    per_expert_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    is_padding_ptr,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    HAS_PADDING: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E

    logits = tl.load(
        gating_ptr + pid * E + offs_e,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    max_l = tl.max(logits, axis=0)

    # Float32 → ascending-sortable bijection
    MIN32 = -2147483648
    logit_bits = logits.to(tl.int32, bitcast=True)
    sign_b = logit_bits >> 31
    key = tl.where(sign_b == 0, logit_bits ^ -1, logit_bits ^ MIN32)
    key = tl.where(valid, key, 0x7FFFFFFF)
    sk64 = key.to(tl.int64) & 0x00000000FFFFFFFF
    packed = (sk64 << 32) | offs_e.to(tl.int64)
    sorted_p = tl.sort(packed, descending=False)

    # Vectorized extraction of ALL sorted elements — no K-loop, no cross-lane reductions
    all_keys = ((sorted_p >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    all_ids = (sorted_p & 0x00000000FFFFFFFF).to(tl.int32)

    # Inverse bijection: recover original logit bits
    sign_k = all_keys >> 31
    all_bits = tl.where(sign_k < 0, all_keys ^ -1, all_keys ^ MIN32)
    all_logits = all_bits.to(tl.float32, bitcast=True)

    # Compute raw_exp for ALL BLOCK_E elements — vectorized, ~2 VALU clocks
    all_raw_exp = tl.math.exp2((all_logits - max_l) * 1.4426950408889634)

    # Sum only top-K for renorm — ONE masked reduction
    top_mask = offs_e < K
    renorm_raw = tl.sum(tl.where(top_mask, all_raw_exp, 0.0), axis=0)
    renorm_raw = tl.where(renorm_raw > 0.0, renorm_raw, 1.0)
    inv_renorm = 1.0 / renorm_raw

    # Load scales for top-K only (masked gather; scale array is tiny → L1 cached)
    all_scales = tl.load(
        per_expert_scale_ptr + all_ids.to(tl.int64),
        mask=top_mask,
        other=1.0,
    ).to(tl.float32)

    # Final weights: vectorized multiply (only top-K will be stored)
    all_weights = (all_raw_exp * inv_renorm * all_scales).to(tl.float32)

    if HAS_PADDING:
        # CUDA-graph padding rows hold stale activations; route them to the
        # -1 skip sentinel so they don't pull real experts' weights.
        is_pad = tl.load(is_padding_ptr + pid).to(tl.int1)
        all_ids = tl.where(is_pad, -1, all_ids)
        all_weights = tl.where(is_pad, 0.0, all_weights)

    # Write results with TWO masked stores — replaces K × 2 serial scalar stores
    base_off = pid * K + offs_e
    tl.store(topk_ids_ptr + base_off, all_ids, mask=top_mask)
    tl.store(topk_weights_ptr + base_off, all_weights, mask=top_mask)


def gemma4_fused_routing_kernel_triton(
    gating_output: torch.Tensor,
    topk: int,
    per_expert_scale: torch.Tensor,
    num_warps: int = 1,
    *,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    gating_output = gating_output.contiguous()
    per_expert_scale = per_expert_scale.contiguous()
    T, E = gating_output.shape
    weights = torch.empty(T, topk, dtype=torch.float32, device=gating_output.device)
    ids = torch.empty(T, topk, dtype=torch.int32, device=gating_output.device)
    BLOCK_E = triton.next_power_of_2(E)
    _gemma4_routing_kernel[(T,)](
        gating_output,
        per_expert_scale,
        weights,
        ids,
        is_padding if is_padding is not None else ids,
        E,
        topk,
        BLOCK_E,
        is_padding is not None,
        num_warps=num_warps,
    )
    return weights, ids
