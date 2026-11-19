# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bias-aware routed top-k without vLLM compiled extensions."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def _topk_routing_kernel(
    logits_ptr,
    bias_ptr,
    padding_ptr,
    weights_ptr,
    indices_ptr,
    source_ptr,
    M,
    E: tl.constexpr,
    K: tl.constexpr,
    SIGMOID: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    scale,
    SCALE_WEIGHTS: tl.constexpr,
    ROWS: tl.constexpr,
    EXPERTS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, EXPERTS)
    valid_rows = rows < M
    valid_experts = cols[None, :] < E
    valid = valid_rows[:, None] & valid_experts
    logits = tl.load(
        logits_ptr + rows[:, None] * E + cols[None, :],
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    if SIGMOID:
        probabilities = 1.0 / (1.0 + tl.exp(-logits))
    else:
        invalid = (logits != logits) | (logits == float("inf"))
        valid_softmax_row = tl.sum(invalid.to(tl.int32), axis=1) == 0
        peak = tl.max(logits, axis=1)
        unnormalized = tl.exp(logits - peak[:, None])
        probabilities = unnormalized / tl.sum(unnormalized, axis=1)[:, None]
        probabilities = tl.where(valid_softmax_row[:, None], probabilities, 0.0)
    probabilities = tl.where(probabilities == probabilities, probabilities, 0.0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=cols < E, other=0).to(tl.float32)
        choice = probabilities + bias[None, :]
    else:
        choice = probabilities
    choice = tl.where(valid_experts, choice, -float("inf"))
    if HAS_PADDING:
        padded = tl.load(padding_ptr + rows, mask=valid_rows, other=False)
    else:
        padded = tl.full((ROWS,), False, tl.int1)
    selected_sum = tl.full((ROWS,), 0.0, tl.float32)
    for rank in range(K):
        _, expert = tl.max(choice, axis=1, return_indices=True)
        value = tl.sum(
            tl.where(cols[None, :] == expert[:, None], probabilities, 0.0), axis=1
        )
        value = tl.where(padded, 0.0, value)
        offset = rows * K + rank
        tl.store(weights_ptr + offset, value, mask=valid_rows)
        tl.store(indices_ptr + offset, tl.where(padded, -1, expert), mask=valid_rows)
        tl.store(source_ptr + offset, rank * M + rows, mask=valid_rows)
        selected_sum += value
        choice = tl.where(cols[None, :] == expert[:, None], -float("inf"), choice)
    if RENORMALIZE or SCALE_WEIGHTS:
        if RENORMALIZE:
            denominator = tl.where(selected_sum > 0.0, selected_sum, 1.0)
        for rank in range(K):
            offset = rows * K + rank
            value = tl.load(weights_ptr + offset, mask=valid_rows, other=0.0)
            if RENORMALIZE:
                value = value / denominator
            if SCALE_WEIGHTS:
                value = value * scale
            tl.store(
                weights_ptr + offset, tl.where(padded, 0.0, value), mask=valid_rows
            )


def _launch_topk_routing(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    is_padding: torch.Tensor | None,
    *,
    sigmoid: bool,
    scale: float,
) -> None:
    assert gating_output.ndim == 2 and topk_weights.ndim == 2
    if gating_output.device.type != "cuda":
        raise NotImplementedError("Extended top-k routing requires a CUDA device")
    m, e = gating_output.shape
    k = topk_weights.shape[1]
    assert gating_output.is_contiguous()
    assert gating_output.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert topk_weights.dtype == torch.float32 and topk_weights.shape == (m, k)
    index_types = (torch.int32, torch.int64)
    if not sigmoid:
        index_types += (torch.uint32,)
    assert topk_indices.dtype in index_types
    assert token_expert_indices.dtype == torch.int32
    assert topk_indices.shape == token_expert_indices.shape == (m, k)
    assert all(
        t.is_contiguous() and t.device == gating_output.device
        for t in (topk_weights, topk_indices, token_expert_indices)
    )
    assert 0 < k <= min(32, e)
    if e > 1024:
        raise NotImplementedError("Top-k routing supports at most 1024 experts")
    if e_score_correction_bias is not None:
        assert e_score_correction_bias.shape == (e,)
        assert e_score_correction_bias.dtype == torch.float32
        assert e_score_correction_bias.is_contiguous()
        assert e_score_correction_bias.device == gating_output.device
    if is_padding is not None:
        assert is_padding.shape == (m,) and is_padding.dtype == torch.bool
        assert is_padding.is_contiguous() and is_padding.device == gating_output.device
    if not m:
        return
    experts = triton.next_power_of_2(e)
    rows = min(8, max(1, 1024 // experts))
    with torch_device_fn.device(gating_output.device):
        _topk_routing_kernel[(triton.cdiv(m, rows),)](
            gating_output,
            e_score_correction_bias,
            is_padding,
            topk_weights,
            topk_indices,
            token_expert_indices,
            m,
            e,
            k,
            sigmoid,
            renormalize,
            e_score_correction_bias is not None,
            is_padding is not None,
            scale,
            scale != 1.0,
            rows,
            experts,
            num_warps=4,
            enable_fp_fusion=False,
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
    routed_scaling_factor: float = 1.0,
    is_padding: torch.Tensor | None = None,
) -> None:
    _launch_topk_routing(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        is_padding,
        sigmoid=True,
        scale=routed_scaling_factor,
    )


@_topk_sigmoid_op.register_fake
def _topk_sigmoid_fake(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize=False,
    e_score_correction_bias=None,
    routed_scaling_factor=1.0,
    is_padding=None,
):
    return None


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    is_padding: torch.Tensor | None = None,
) -> None:
    """Route with sigmoid scores, then scale the optionally normalized weights.

    Bias affects expert selection only. Padded rows have zero weights and
    expert ID -1, with valid expanded token row indices.
    """
    _topk_sigmoid_op(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        routed_scaling_factor,
        is_padding,
    )
