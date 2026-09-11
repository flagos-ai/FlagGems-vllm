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

"""PyTorch references used only by mHC tests and benchmarks."""

from __future__ import annotations

import torch

_CG_EPS = 1e-10


def hc_head_fused_kernel_ref(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> torch.Tensor:
    """Independent reference for the fused head operation."""
    if hs_flat.shape[0] == 0:
        return out
    x = hs_flat.reshape(hs_flat.shape[0], hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps
    result = torch.sum(pre_mix.unsqueeze(-1) * hs_flat.to(torch.float32), dim=1).to(
        out.dtype
    )
    out.copy_(result)
    return out


def mhc_split_sinkhorn_torch_ref(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent reference for split, activation, and Sinkhorn."""
    outer_shape = mixes.shape[:-1]
    mix_hc = (2 + hc_mult) * hc_mult
    assert mixes.shape[-1] == mix_hc

    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].view(*outer_shape, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)

    row_max = comb.max(dim=-1, keepdim=True).values
    comb = (comb - row_max).exp()
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def mhc_bwd_ref(
    out: torch.Tensor,
    dout: torch.Tensor,
    cg_iters: int | None = None,
) -> torch.Tensor:
    """Independent implicit-CG Sinkhorn backward reference."""
    _, n_stream, _ = out.shape
    if cg_iters is None:
        cg_iters = 2 * n_stream

    matrix = out.float()
    matrix_grad = dout.float()
    product = matrix * matrix_grad
    right_row = product.sum(dim=-1)
    right_column = product.sum(dim=-2)
    solution_row = torch.zeros_like(right_row)
    solution_column = torch.zeros_like(right_column)

    def matvec(row_input, column_input):
        row_output = (matrix * column_input.unsqueeze(-2)).sum(dim=-1) + row_input
        column_output = (matrix * row_input.unsqueeze(-1)).sum(dim=-2) + column_input
        return row_output, column_output

    residual_row, residual_column = right_row.clone(), right_column.clone()
    direction_row, direction_column = residual_row.clone(), residual_column.clone()
    residual_norm = (residual_row.square() + residual_column.square()).sum(dim=-1)

    for _ in range(cg_iters):
        product_row, product_column = matvec(direction_row, direction_column)
        direction_product = (
            direction_row * product_row + direction_column * product_column
        ).sum(dim=-1)
        alpha = (residual_norm / (direction_product + _CG_EPS)).unsqueeze(-1)
        solution_row += alpha * direction_row
        solution_column += alpha * direction_column
        residual_row -= alpha * product_row
        residual_column -= alpha * product_column
        next_residual_norm = (residual_row.square() + residual_column.square()).sum(
            dim=-1
        )
        beta = (next_residual_norm / (residual_norm + _CG_EPS)).unsqueeze(-1)
        direction_row = residual_row + beta * direction_row
        direction_column = residual_column + beta * direction_column
        residual_norm = next_residual_norm

    return (
        matrix_grad - solution_row.unsqueeze(-1) - solution_column.unsqueeze(-2)
    ) * matrix


def sinkhorn_forward(
    logits: torch.Tensor, iters: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent Sinkhorn forward reference."""
    exponent = torch.exp(logits)
    result = exponent.clone()
    for _ in range(iters):
        result = result / result.sum(-2, keepdim=True)
        result = result / result.sum(-1, keepdim=True)
    return result, exponent
