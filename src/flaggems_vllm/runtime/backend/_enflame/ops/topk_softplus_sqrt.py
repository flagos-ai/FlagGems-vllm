# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from the vLLM project (https://github.com/vllm-project/vllm).
# Source: vllm/model_executor/layers/fused_moe/topk_softplus_sqrt_kernels.cu
#
# This Triton implementation is based on the CUDA kernel from vLLM 0.20.0.
# The kernel fuses softplus, sqrt, top-k selection, and optional renormalization
# for MoE gating in models like DeepSeek-V3/V4.

"""TopK Softplus-Sqrt gating kernel in Triton.

Optimized v27: num_warps=1 + all v19-v26 wins.
Key insight: For 256 experts, CUDA uses exactly 1 warp (32 threads) per row,
with each thread holding 8 elements. Using num_warps=4 adds warp scheduling
overhead without helping the 256-element reduction. Combining num_warps=1
(matching CUDA's single-warp-per-row) with tensor caching (v19), score-arithmetic
weight extraction (v20), and the max+compare index recovery (v26) should
minimize overhead.

Eliminates the store-load-store pattern for renormalization by storing weights
during the loop and re-reading with scale at the end.

GCU row-batch path (enflame): the 1-token-per-program grid above is
dispatch/serial-chain bound (~0.19 us/token at 256 experts). For
num_tokens >= 128 the host dispatches `_fused_topk_rowbatch_kernel`
([BLOCK_M, BLOCK_E] tile, one program per BLOCK_M rows, weights kept in
registers through the k rounds) with an empirical (BLOCK_M, num_warps)
ladder -- see `_rowbatch_cfg`. Outputs are bitwise identical to the 1-D
kernel.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255. The kernels use one program
# per token row, so token counts above 65535 must be spread over grid.y/grid.z
# by _token_grid(); the kernels recombine (pid_x, pid_y, pid_z) into a flat
# global token id via tl.num_programs, keeping token_expert_indices global.
_GRID_X_MAX = 65535
_GRID_Y_MAX = 255


def _token_grid(num_tokens):
    """Spread ``num_tokens`` logical rows over a 3-D grid within GCU limits."""
    if num_tokens <= _GRID_X_MAX:
        return (num_tokens, 1, 1)

    grid_y = triton.cdiv(num_tokens, _GRID_X_MAX)
    if grid_y <= _GRID_Y_MAX:
        return (triton.cdiv(num_tokens, grid_y), grid_y, 1)

    grid_z = triton.cdiv(num_tokens, _GRID_X_MAX * _GRID_Y_MAX)
    grid_y = triton.cdiv(num_tokens, _GRID_X_MAX * grid_z)
    return (triton.cdiv(num_tokens, grid_y * grid_z), grid_y, grid_z)


_ROW_BATCH_MIN_TOKENS = 128


def _rowbatch_cfg(num_tokens, block_e, narrow_dtype):
    """Pick ``(BLOCK_M, num_warps)`` for the row-batch kernel, else ``None``.

    Empirical ladder on GCU (interleaved do_bench median, E=256/topk=6):
    the 1-token-per-program kernel is dispatch/serial-chain bound
    (~0.19 us/token); row-batching amortizes both, e.g. T=512
    98->30 us (BM32/nw2), T=1024 195->36 (BM64/nw4), T=4096 776->87
    (~8.9x). Below 128 tokens the 1-D kernel stays at the launch floor.

    num_warps>2 is only enabled for 2-byte gating dtypes with
    BLOCK_E <= 256: nw=4 silently miscompiles the k-round argmax chain
    outside that domain (P10 row-reduce x warp family -- BLOCK_E>=512 at
    bf16/fp16, BLOCK_E>=256 already at fp32; see compiler_problems/P10_.../
    sources/flaggems_vllm_topk_softplus_sqrt_wide_e_nw4_rowreduce.md).
    Every shipped (dtype, BLOCK_E, BLOCK_M, num_warps) combo is
    bitwise-verified against the 1-D kernel. BLOCK_E >= 2048 is
    unverified territory and falls back to the 1-D kernel.
    """
    if num_tokens < _ROW_BATCH_MIN_TOKENS or block_e >= 2048:
        return None
    if block_e >= 512 or not narrow_dtype:
        return (16, 2) if num_tokens < 512 else (32, 2)
    if num_tokens >= 1024:
        return (64, 4)
    if num_tokens >= 512:
        return (32, 2)
    return (16, 2)


@triton.jit
def _fused_topk_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    e_score_correction_bias_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    grid_x = tl.num_programs(0)
    grid_xy = grid_x * tl.num_programs(1)
    pid = tl.program_id(0) + tl.program_id(1) * grid_x + tl.program_id(2) * grid_xy
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    emask = expert_offsets < num_experts

    row_base = pid * num_experts
    x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(
        tl.float32
    )

    # Fused softplus + sqrt
    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    raw = tl.sqrt(x)

    # Scores for top-k selection (with optional bias)
    if HAS_BIAS:
        bias = tl.load(
            e_score_correction_bias_ptr + expert_offsets, mask=emask, other=0.0
        ).to(tl.float32)
        scores = raw + bias
    else:
        scores = raw
    scores = tl.where(emask, scores, -float("inf"))

    out_base = pid * topk
    weight_sum = 0.0

    for k_idx in tl.static_range(topk):
        max_score = tl.max(scores, axis=0)
        is_max = scores == max_score
        match_priority = tl.where(is_max, BLOCK_E - expert_offsets, 0)
        best_slot = BLOCK_E - tl.max(match_priority, axis=0)
        eidx = best_slot.to(tl.int32)

        if HAS_BIAS:
            bias_at_eidx = tl.load(e_score_correction_bias_ptr + eidx)
            w = max_score - bias_at_eidx
        else:
            w = max_score

        weight_sum += w
        tl.store(topk_weights_ptr + out_base + k_idx, w)
        tl.store(topk_indices_ptr + out_base + k_idx, eidx)
        tl.store(
            token_expert_indices_ptr + out_base + k_idx,
            (pid * topk + k_idx).to(tl.int32),
        )

        # Zero out winner
        scores = tl.where(expert_offsets == eidx, -float("inf"), scores)

    # Renormalize: re-read weights and apply scale
    if renormalize:
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    else:
        scale = routed_scaling_factor

    for k_idx in tl.static_range(topk):
        w = tl.load(topk_weights_ptr + out_base + k_idx)
        tl.store(topk_weights_ptr + out_base + k_idx, w * scale)


@triton.jit
def _fused_topk_rowbatch_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    e_score_correction_bias_ptr,
    num_tokens,
    num_blocks,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
):
    """Row-batch variant: one program covers BLOCK_M token rows.

    Per-row math and evaluation order are identical to _fused_topk_kernel
    (outputs are bitwise equal); only the program<->token mapping and the
    store epilogue differ -- weights/indices stay in registers through the
    k rounds and are written once per tile, so the per-token store/reload
    traffic of the 1-D kernel disappears and the serial k-round chain is
    amortized over BLOCK_M rows.
    """
    grid_x = tl.num_programs(0)
    grid_xy = grid_x * tl.num_programs(1)
    pid = tl.program_id(0) + tl.program_id(1) * grid_x + tl.program_id(2) * grid_xy
    if pid >= num_blocks:
        return

    row0 = pid * BLOCK_M
    rows = row0 + tl.arange(0, BLOCK_M)
    offs = tl.arange(0, BLOCK_E)
    koffs = tl.arange(0, BLOCK_K)
    kmask = koffs < topk
    emask = offs < num_experts

    if EVEN_ROWS:
        if num_experts == BLOCK_E:
            x = tl.load(gating_ptr + rows[:, None] * num_experts + offs[None, :])
        else:
            x = tl.load(
                gating_ptr + rows[:, None] * num_experts + offs[None, :],
                mask=emask[None, :],
                other=0.0,
            )
    else:
        rmask = rows < num_tokens
        x = tl.load(
            gating_ptr + rows[:, None] * num_experts + offs[None, :],
            mask=rmask[:, None] & emask[None, :],
            other=0.0,
        )
    x = x.to(tl.float32)

    # Fused softplus + sqrt
    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    raw = tl.sqrt(x)

    # Scores for top-k selection (with optional bias)
    if HAS_BIAS:
        bias = tl.load(e_score_correction_bias_ptr + offs, mask=emask, other=0.0).to(
            tl.float32
        )
        scores = raw + bias[None, :]
    else:
        scores = raw

    if EVEN_ROWS:
        if num_experts != BLOCK_E:
            scores = tl.where(emask[None, :], scores, -float("inf"))
    else:
        scores = tl.where(emask[None, :] & rmask[:, None], scores, -float("inf"))

    weight_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    weights = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    idxs = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)

    for k in tl.static_range(topk):
        max_score = tl.max(scores, axis=1)
        is_max = scores == max_score[:, None]
        match_priority = tl.where(is_max, BLOCK_E - offs, 0)
        best_slot = BLOCK_E - tl.max(match_priority, axis=1)
        eidx = best_slot.to(tl.int32)

        if HAS_BIAS:
            bias_at_eidx = tl.load(e_score_correction_bias_ptr + eidx)
            w = max_score - bias_at_eidx
        else:
            w = max_score

        weight_sum += w
        sel = koffs[None, :] == k
        weights = tl.where(sel, w[:, None], weights)
        idxs = tl.where(sel, eidx[:, None], idxs)

        # Zero out winner
        scores = tl.where(offs[None, :] == eidx[:, None], -float("inf"), scores)

    # Renormalize: weights are still in registers, single burst store
    if renormalize:
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    else:
        scale = routed_scaling_factor
    weights = weights * scale[:, None]

    out_off = rows[:, None] * topk + koffs[None, :]
    if EVEN_ROWS:
        omask = kmask[None, :]
    else:
        omask = kmask[None, :] & rmask[:, None]
    tl.store(topk_weights_ptr + out_off, weights, mask=omask)
    tl.store(topk_indices_ptr + out_off, idxs, mask=omask)
    tei = (rows[:, None] * topk + koffs[None, :]).to(tl.int32)
    tl.store(token_expert_indices_ptr + out_off, tei, mask=omask)


@triton.jit
def _hash_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    e_score_correction_bias_ptr,
    input_tokens_ptr,
    hash_indices_table_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Hash mode: expert indices come from lookup table."""
    grid_x = tl.num_programs(0)
    grid_xy = grid_x * tl.num_programs(1)
    pid = tl.program_id(0) + tl.program_id(1) * grid_x + tl.program_id(2) * grid_xy
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    emask = expert_offsets < num_experts

    row_base = pid * num_experts
    x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(
        tl.float32
    )

    # Fused softplus + sqrt
    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    x = tl.sqrt(x)

    # Get expert indices from lookup table
    token_id = tl.load(input_tokens_ptr + pid)
    k_offsets = tl.arange(0, BLOCK_K)
    kmask = k_offsets < topk
    expert_ids = tl.load(
        hash_indices_table_ptr + token_id * topk + k_offsets, mask=kmask, other=0
    )

    # Gather weights for each selected expert
    weight_sum = 0.0
    weights = tl.zeros([BLOCK_K], dtype=tl.float32)

    for k_idx in tl.static_range(topk):
        eidx = tl.sum(tl.where(k_offsets == k_idx, expert_ids, 0))
        w = tl.sum(tl.where(expert_offsets == eidx, x, 0.0))
        weight_sum += w
        weights = tl.where(k_offsets == k_idx, w, weights)

    # Apply renormalization + scaling
    if renormalize:
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    else:
        scale = routed_scaling_factor
    weights = weights * scale

    # Single burst store
    out_base = pid * topk
    tl.store(topk_weights_ptr + out_base + k_offsets, weights, mask=kmask)
    tl.store(topk_indices_ptr + out_base + k_offsets, expert_ids, mask=kmask)
    tei = (pid * topk + k_offsets).to(tl.int32)
    tl.store(token_expert_indices_ptr + out_base + k_offsets, tei, mask=kmask)


def topk_softplus_sqrt(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    correction_bias=None,
    input_ids=None,
    tid2eid=None,
):
    """Fused topk + softplus + sqrt kernel for MoE gating.

    Interface aligned with vLLM CUDA operator:
        void topk_softplus_sqrt(Tensor& topk_weights, Tensor& topk_indices,
            Tensor& token_expert_indices, Tensor& gating_output,
            bool renormalize, double routed_scaling_factor,
            const c10::optional<Tensor>& correction_bias,
            const c10::optional<Tensor>& input_ids,
            const c10::optional<Tensor>& tid2eid);

    Args:
        topk_weights: Output tensor [num_tokens, topk], dtype float32
        topk_indices: Output tensor [num_tokens, topk], dtype int32
        token_expert_indices: Output tensor [num_tokens, topk], dtype int32
        gating_output: Gating logits [num_tokens, num_experts]
        renormalize: Whether to renormalize weights
        routed_scaling_factor: Scaling factor for final weights
        correction_bias: Optional bias for expert scores [num_experts]
        input_ids: Token IDs for hash mode [num_tokens]
        tid2eid: Hash table mapping tokens to expert indices
    """
    logger.debug("GEMS TOPK_SOFTPLUS_SQRT")
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]

    if num_tokens == 0:
        return

    BLOCK_E = triton.next_power_of_2(num_experts)

    if input_ids is not None and tid2eid is not None:
        BLOCK_K = triton.next_power_of_2(topk)
        grid = _token_grid(num_tokens)
        _hash_kernel[grid](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            correction_bias if correction_bias is not None else gating_output,
            input_ids,
            tid2eid,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            HAS_BIAS=correction_bias is not None,
            BLOCK_E=BLOCK_E,
            BLOCK_K=BLOCK_K,
            num_warps=1,
            num_stages=1,
        )
        return

    narrow_dtype = gating_output.dtype in (torch.bfloat16, torch.float16)
    cfg = _rowbatch_cfg(num_tokens, BLOCK_E, narrow_dtype)

    if cfg is not None:
        block_m, warps = cfg
        num_blocks = triton.cdiv(num_tokens, block_m)
        _fused_topk_rowbatch_kernel[_token_grid(num_blocks)](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            correction_bias if correction_bias is not None else gating_output,
            num_tokens=num_tokens,
            num_blocks=num_blocks,
            num_experts=num_experts,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            HAS_BIAS=correction_bias is not None,
            BLOCK_E=BLOCK_E,
            BLOCK_K=max(2, triton.next_power_of_2(topk)),
            BLOCK_M=block_m,
            EVEN_ROWS=(num_tokens % block_m == 0),
            num_warps=warps,
            num_stages=1,
        )
        return

    grid = _token_grid(num_tokens)
    _fused_topk_kernel[grid](
        gating_output,
        topk_weights,
        topk_indices,
        token_expert_indices,
        correction_bias if correction_bias is not None else gating_output,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        HAS_BIAS=correction_bias is not None,
        BLOCK_E=BLOCK_E,
        num_warps=1,
        num_stages=1,
    )


# ---------------------------------------------------------------------------
# vLLM torch.ops registration.
#
# On the GCU there is no CUDA driver, so vLLM's `_moe_C` extension never loads
# and `torch.ops._moe_C.topk_softplus_sqrt` stays unregistered: vLLM's
# `topk_hash_softplus_sqrt` wrapper (used by every DeepSeek-V3/V4-style router
# path and by this repo's vs-vLLM tests/benchmarks) then fails with
# `AttributeError: '_OpNamespace' '_moe_C' object has no attribute
# 'topk_softplus_sqrt'`. Register this backend's Triton implementation under
# the same qualified name so the wrapper resolves on GCU. Hosts where the
# native CUDA op is already registered (real CUDA boxes) skip the override.
# ---------------------------------------------------------------------------
if not hasattr(torch.ops._moe_C, "topk_softplus_sqrt"):
    _moe_c_lib = torch.library.Library("_moe_C", "FRAGMENT")
    _moe_c_lib.define(
        "topk_softplus_sqrt(Tensor(a!) topk_weights, Tensor(b!) topk_indices, "
        "Tensor(c!) token_expert_indices, Tensor gating_output, bool renormalize, "
        "float routed_scaling_factor, Tensor? correction_bias, Tensor? input_ids, "
        "Tensor? tid2eid) -> ()"
    )
    _moe_c_lib.impl("topk_softplus_sqrt", topk_softplus_sqrt)
