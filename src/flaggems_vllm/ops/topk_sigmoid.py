# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bias-aware sigmoid routing without vLLM compiled extensions."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


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
    ROWS: tl.constexpr,
    EXPERTS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, EXPERTS)
    valid = (rows[:, None] < M) & (cols[None, :] < E)
    logits = tl.load(
        logits_ptr + rows[:, None] * E + cols[None, :], mask=valid, other=0
    ).to(tl.float32)
    probabilities = 1.0 / (1.0 + tl.exp(-logits))
    # Match native graph-padding semantics: NaN sigmoid becomes zero.
    probabilities = tl.where(probabilities == probabilities, probabilities, 0.0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=cols < E, other=0).to(tl.float32)
        choice = probabilities + bias[None, :]
    else:
        choice = probabilities
    choice = tl.where(cols[None, :] < E, choice, -float("inf"))
    selected_sum = tl.full((ROWS,), 0.0, tl.float32)
    for rank in range(K):
        _, expert = tl.max(choice, axis=1, return_indices=True)
        value = tl.sum(
            tl.where(cols[None, :] == expert[:, None], probabilities, 0.0), axis=1
        )
        offset = rows * K + rank
        tl.store(weights_ptr + offset, value, mask=rows < M)
        tl.store(indices_ptr + offset, expert, mask=rows < M)
        tl.store(source_ptr + offset, rank * M + rows, mask=rows < M)
        selected_sum += value
        choice = tl.where(cols[None, :] == expert[:, None], -float("inf"), choice)
    if RENORMALIZE:
        denominator = tl.where(selected_sum > 0.0, selected_sum, 1.0)
        for rank in range(K):
            offset = rows * K + rank
            value = tl.load(weights_ptr + offset, mask=rows < M, other=0.0)
            tl.store(weights_ptr + offset, value / denominator, mask=rows < M)


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
    if gating_output.device.type != "cuda":
        raise NotImplementedError("Sigmoid routing requires a CUDA device")
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
    experts = triton.next_power_of_2(e)
    rows = min(8, max(1, 1024 // experts))
    with torch_device_fn.device(gating_output.device):
        _topk_sigmoid_kernel[(triton.cdiv(m, rows),)](
            gating_output,
            e_score_correction_bias,
            topk_weights,
            topk_indices,
            token_expert_indices,
            m,
            e,
            k,
            renormalize,
            e_score_correction_bias is not None,
            rows,
            experts,
            num_warps=4,
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
