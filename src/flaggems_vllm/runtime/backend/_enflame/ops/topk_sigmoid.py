# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bias-aware sigmoid routing without vLLM compiled extensions."""

import os

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GRID_X_MAX = 65535
_GRID_Y_MAX = 255


def _row_grid(m):
    """Spread ``m`` logical rows over a 3-D grid within GCU launch limits.

    The kernel recombines (pid_x, pid_y, pid_z) into a flat row id via the
    ``GRID_X``/``GRID_XY`` runtime arguments, so any decomposition with
    ``grid_x * grid_y * grid_z >= m`` is correct; rows beyond ``m`` are
    masked out by the in-kernel ``row < M`` guard.
    """
    if m <= _GRID_X_MAX:
        return (m, 1, 1)

    grid_y = triton.cdiv(m, _GRID_X_MAX)
    if grid_y <= _GRID_Y_MAX:
        return (triton.cdiv(m, grid_y), grid_y, 1)

    grid_z = triton.cdiv(m, _GRID_X_MAX * _GRID_Y_MAX)
    grid_y = triton.cdiv(m, _GRID_X_MAX * grid_z)
    return (triton.cdiv(m, grid_y * grid_z), grid_y, grid_z)


# Dispatch thresholds (measured on GCU, e=128 fp32, do_bench median):
# - row-per-program packed-key kernel wins while the device is latency
#   bound by the per-row selection chain (m=1: 7.5us vs 37.8us baseline,
#   m=128: 28.1us vs ~76us);
# - beyond ~128 rows the device saturates and the int64 payload costs
#   throughput, so rows are batched into 2-D tiles that share one argmax
#   tree per rank (m=4096: 369us vs 799us, m=8192: 686us vs 1604us).
_SMALL_M_MAX = 128
_ROWBLOCK_M1_MAX = 2048
_ROWBLOCK_L1 = 16
_ROWBLOCK_L2 = 32
# P10 (cross-warp reduction miscompile family): [*,256]-wide reduction
# tiles corrupt under multi-warp launches on this toolchain — the int64
# packed-key max mis-sorts intermittently at num_warps=4, the 2-D fp32
# argmax mis-selects deterministically, and the sibling topk_softmax bias
# arm was broken even at num_warps=2. Single-warp launches are
# structurally immune, so expert counts above 128 are pinned to
# num_warps=1 (verified clean over 30-iteration soak runs).
_NW4_EXPERTS_MAX = 128


def _num_warps(experts):
    return 4 if experts <= _NW4_EXPERTS_MAX else 1


@libentry()
@triton.jit
def _topk_sigmoid_kernel(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    indices_ptr,
    source_ptr,
    M,
    E: tl.constexpr,
    K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EXPERTS: tl.constexpr,
    KR: tl.constexpr,
    GRID_X,
    GRID_XY,
):
    # One token row per program, 1-D vector addressing. The selection loop
    # runs on a single int64 key per expert (monotonic fp32 order map in
    # the high bits, tie-break-left column index in the low bits), so each
    # of the K rounds costs one plain tl.max tree instead of a (value,
    # index) pair tree; selected weights are re-gathered from logits after
    # the loop (same sigmoid chain => bit-identical values).
    row = tl.program_id(0) + tl.program_id(1) * GRID_X + tl.program_id(2) * GRID_XY
    if row < M:
        cols = tl.arange(0, EXPERTS)
        valid = cols < E
        logits = tl.load(logits_ptr + row * E + cols, mask=valid, other=0).to(
            tl.float32
        )
        probabilities = tl.sigmoid(logits)
        # Match native graph-padding semantics: NaN sigmoid becomes zero.
        probabilities = tl.where(probabilities == probabilities, probabilities, 0.0)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + cols, mask=valid, other=0).to(tl.float32)
            choice = probabilities + bias
        else:
            choice = probabilities
        # +0.0 folds -0.0 onto +0.0 so the bit-order key below is exact.
        choice = tl.where(valid, choice + 0.0, -float("inf"))
        key = choice.to(tl.int32, bitcast=True)
        key = tl.where(key >= 0, key, ~key ^ (-2147483648))
        packed = (key.to(tl.int64) << 8) | (255 - cols).to(tl.int64)
        packed = tl.where(valid, packed, -(1 << 63))
        ranks = tl.arange(0, KR)
        top_experts = tl.zeros((KR,), tl.int32)
        for rank in range(K):
            best = tl.max(packed, axis=0)
            expert = (255 - (best & 255)).to(tl.int32)
            top_experts = tl.where(ranks == rank, expert, top_experts)
            packed = tl.where(cols == expert, -(1 << 63), packed)
        keep = ranks < K
        gathered = tl.load(logits_ptr + row * E + top_experts, mask=keep, other=0).to(
            tl.float32
        )
        top_weights = tl.sigmoid(gathered)
        top_weights = tl.where(top_weights == top_weights, top_weights, 0.0)
        top_weights = tl.where(keep, top_weights, 0.0)
        if RENORMALIZE:
            denominator = tl.sum(top_weights, axis=0, keep_dims=True)
            denominator = tl.where(denominator > 0.0, denominator, 1.0)
            top_weights = top_weights / denominator
        offsets = row * K + ranks
        tl.store(weights_ptr + offsets, top_weights, mask=keep)
        tl.store(indices_ptr + offsets, top_experts, mask=keep)
        tl.store(source_ptr + offsets, ranks * M + row, mask=keep)


@libentry()
@triton.jit
def _topk_sigmoid_rowblock_kernel(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    indices_ptr,
    source_ptr,
    M,
    E: tl.constexpr,
    K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EXPERTS: tl.constexpr,
    KR: tl.constexpr,
    RBLOCK: tl.constexpr,
    GRID_X,
    GRID_XY,
):
    # RBLOCK rows per program as a [RBLOCK, EXPERTS] tile: the per-rank
    # tl.max(axis=1) tree and the elimination select are amortized across
    # the whole tile, which dominates once the device is saturated by the
    # row count. Selection uses the fp32 pair-tree (tie-break-left); the
    # weights are gathered back from logits after the loop so the per-rank
    # probability extraction tree disappears.
    block = tl.program_id(0) + tl.program_id(1) * GRID_X + tl.program_id(2) * GRID_XY
    rows = block * RBLOCK + tl.arange(0, RBLOCK)
    row_ok = rows < M
    cols = tl.arange(0, EXPERTS)
    valid = row_ok[:, None] & (cols[None, :] < E)
    logits = tl.load(
        logits_ptr + rows[:, None] * E + cols[None, :], mask=valid, other=0
    ).to(tl.float32)
    probabilities = tl.sigmoid(logits)
    probabilities = tl.where(probabilities == probabilities, probabilities, 0.0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=cols < E, other=0).to(tl.float32)
        choice = probabilities + bias[None, :]
    else:
        choice = probabilities
    choice = tl.where(valid, choice, -float("inf"))
    ranks = tl.arange(0, KR)
    top_experts = tl.zeros((RBLOCK, KR), tl.int32)
    for rank in range(K):
        _, expert = tl.max(choice, axis=1, return_indices=True)
        top_experts = tl.where(
            (ranks[None, :] == rank) & row_ok[:, None], expert[:, None], top_experts
        )
        choice = tl.where(cols[None, :] == expert[:, None], -float("inf"), choice)
    keep = ranks[None, :] < K
    omask = row_ok[:, None] & keep
    gathered = tl.load(
        logits_ptr + rows[:, None] * E + top_experts, mask=omask, other=0
    ).to(tl.float32)
    top_weights = tl.sigmoid(gathered)
    top_weights = tl.where(top_weights == top_weights, top_weights, 0.0)
    top_weights = tl.where(omask, top_weights, 0.0)
    if RENORMALIZE:
        denominator = tl.sum(top_weights, axis=1, keep_dims=True)
        denominator = tl.where(denominator > 0.0, denominator, 1.0)
        top_weights = top_weights / denominator
    offsets = rows[:, None] * K + ranks[None, :]
    tl.store(weights_ptr + offsets, top_weights, mask=omask)
    tl.store(indices_ptr + offsets, top_experts, mask=omask)
    tl.store(source_ptr + offsets, ranks[None, :] * M + rows[:, None], mask=omask)


def _gcu_int64_enabled():
    return os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "").lower() not in (
        "",
        "0",
        "false",
        "off",
        "no",
    )


@torch.library.custom_op(
    "flaggems_vllm::topk_sigmoid",
    mutates_args=("topk_weights", "topk_indices", "token_expert_indices"),
)
def _topk_sigmoid_op(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    assert gating_output.ndim == 2 and topk_weights.ndim == 2
    if gating_output.device.type != runtime.device.name:
        raise NotImplementedError(
            "Sigmoid routing requires a %s device" % runtime.device.name
        )
    m, e = gating_output.shape
    k = topk_weights.shape[1]
    assert gating_output.ndim == 2 and gating_output.is_contiguous()
    assert gating_output.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert topk_weights.dtype == torch.float32 and topk_weights.shape == (m, k)
    assert topk_indices.dtype in (torch.int32, torch.int64)
    assert token_expert_indices.dtype == torch.int32
    assert topk_indices.shape == token_expert_indices.shape == (m, k)
    assert all(
        t.is_contiguous() and t.device == gating_output.device
        for t in (topk_weights, topk_indices, token_expert_indices)
    )
    assert 0 < k <= min(32, e)
    if e > 256:
        raise NotImplementedError("Sigmoid routing supports at most 256 experts")
    if e_score_correction_bias is not None:
        assert e_score_correction_bias.shape == (e,)
        assert e_score_correction_bias.dtype == torch.float32
        assert e_score_correction_bias.is_contiguous()
        assert e_score_correction_bias.device == gating_output.device
    if not m:
        return
    # torch_gcu narrows device int64 tensors to int32 physical storage while
    # the tensor still reports int64 ("GCU not support Long use Int replace");
    # a Triton store at the reported int64 width would stride past the real
    # slots and leave interleaved garbage. Expert ids always fit in int32, so
    # hand the kernel a same-shape/same-stride int32 view of the same storage
    # (element offset == word offset, zero copy). With the int64 env switch on
    # (real 8-byte storage) the tensor is passed through untouched.
    indices_out = topk_indices
    if topk_indices.dtype == torch.int64 and not _gcu_int64_enabled():
        indices_out = torch.empty(0, dtype=torch.int32, device=topk_indices.device)
        indices_out.set_(
            topk_indices.untyped_storage(),
            topk_indices.storage_offset(),
            topk_indices.shape,
            topk_indices.stride(),
        )
    experts = triton.next_power_of_2(e)
    kr = triton.next_power_of_2(k)
    num_warps = _num_warps(experts)
    with torch_device_fn.device(gating_output.device):
        if m <= _SMALL_M_MAX:
            # Latency regime: one row per program, packed-key selection
            # (single int64 tl.max tree per rank).
            grid_x, grid_y, grid_z = _row_grid(m)
            _topk_sigmoid_kernel[(grid_x, grid_y, grid_z)](
                gating_output,
                e_score_correction_bias,
                topk_weights,
                indices_out,
                token_expert_indices,
                m,
                e,
                k,
                renormalize,
                e_score_correction_bias is not None,
                experts,
                kr,
                grid_x,
                grid_x * grid_y,
                num_warps=num_warps,
                enable_fp_fusion=False,
            )
        else:
            # Throughput regime: batch RBLOCK rows per program so the
            # per-rank argmax tree is shared across the tile.
            rblock = _ROWBLOCK_L1 if m <= _ROWBLOCK_M1_MAX else _ROWBLOCK_L2
            grid_x, grid_y, grid_z = _row_grid(triton.cdiv(m, rblock))
            _topk_sigmoid_rowblock_kernel[(grid_x, grid_y, grid_z)](
                gating_output,
                e_score_correction_bias,
                topk_weights,
                indices_out,
                token_expert_indices,
                m,
                e,
                k,
                renormalize,
                e_score_correction_bias is not None,
                experts,
                kr,
                rblock,
                grid_x,
                grid_x * grid_y,
                num_warps=num_warps,
                enable_fp_fusion=False,
            )


@_topk_sigmoid_op.register_fake
def _topk_sigmoid_fake(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize=False,
    e_score_correction_bias=None,
):
    return None


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    _topk_sigmoid_op(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
    )
