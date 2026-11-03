# SPDX-License-Identifier: Apache-2.0
"""Torch baselines shared by numerical Qwen4 benchmarks only."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NULL_BLOCK_ID = -1


def _reference_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, dim = residual.shape
    hidden_size = dim // hc_count
    weight = norm_weight.float()
    combined = (
        residual.float().view(rows, hc_count, hidden_size)
        + block_output.float()[:, None, :]
        * (2.0 * torch.sigmoid(injection_logits.float() / hc_count))[:, :, None]
    ).to(residual.dtype)
    grouped = combined.float().view(rows, hc_count, hidden_size)
    inv_rms = torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    if weight.numel() == hidden_size:
        affine = weight.view(1, 1, hidden_size)
    else:
        affine = weight.view(1, hc_count, hidden_size)
    normalized = (grouped * inv_rms * (1.0 + affine)).flatten(1).to(residual.dtype)
    return combined.flatten(1), normalized


def _grouped_norm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (
        x_float
        * torch.rsqrt(variance + eps)
        * (1.0 + weight.float().view(1, *x.shape[1:]))
    ).to(x.dtype)


def _gate_norm_reference(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = key.shape[-1]
    norm_key = _grouped_norm_reference(key, key_weight, eps)
    norm_query = _grouped_norm_reference(query, query_weight, eps)
    gate = (norm_key * norm_query).sum(dim=-1, keepdim=True) / math.sqrt(hidden_size)
    gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1.0e-6).sqrt())
    gated = gate * value.unsqueeze(1)
    normalized = _grouped_norm_reference(gated, conv_weight, eps)
    return gated, normalized


def sparse_reference(
    q, k, v, indices, table, requests, *, gate=None, split_workspace=None
):
    page_size = k.shape[1]
    pages = indices // page_size
    valid = (
        (indices >= 0)
        & (pages < table.shape[1])
        & (requests[:, None] >= 0)
        & (requests[:, None] < table.shape[0])
    )
    physical = table[
        requests.clamp(0, table.shape[0] - 1)[:, None],
        pages.clamp(0, table.shape[1] - 1),
    ].long()
    valid &= (physical >= 0) & (physical < k.shape[0])
    offset = indices.remainder(page_size)
    keys = (
        k[physical.clamp(0, k.shape[0] - 1), offset]
        .repeat_interleave(q.shape[1] // k.shape[2], dim=2)
        .transpose(1, 2)
    )
    values = (
        v[physical.clamp(0, v.shape[0] - 1), offset]
        .repeat_interleave(q.shape[1] // v.shape[2], dim=2)
        .transpose(1, 2)
    )
    scores = (q.float().unsqueeze(2) * keys.float()).sum(-1) * q.shape[2] ** -0.5
    scores.masked_fill_(~valid[:, None], -float("inf"))
    probabilities = scores.softmax(-1).nan_to_num()
    output = (probabilities.unsqueeze(-1) * values.float()).sum(2).to(q.dtype)
    return output if gate is None else output * torch.sigmoid(gate)


def prefill_conv_reference(
    x,
    output,
    state,
    weight,
    starts,
    indices,
    has_initial,
    *,
    num_prefills,
    max_len,
    state_len,
    kernel_width,
    dilation,
    null_block_id,
):
    # Benchmark uses one fixed full prefill request, with reset-free zero initial state.
    history = torch.cat(
        (
            torch.zeros((1, x.shape[1], state_len), device=x.device, dtype=x.dtype),
            x.T.unsqueeze(0),
        ),
        dim=-1,
    )
    result = (
        F.silu(
            F.conv1d(history, weight.unsqueeze(1), groups=x.shape[1], dilation=dilation)
        )
        .squeeze(0)
        .T
    )
    output.copy_(result)
    state[1, :, :state_len].copy_(history[0, :, -state_len:])
    return True
