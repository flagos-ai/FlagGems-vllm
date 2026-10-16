# SPDX-License-Identifier: Apache-2.0
"""Rounded HC combine/norm reference and graph replay checks."""

from __future__ import annotations

import pytest
import torch

from flaggems_vllm.ops.qwen4.hyperconnection import (
    can_use_hc_combine_norm_triton,
    qwen4_grouped_gemma_rmsnorm,
    qwen4_hc_combine_norm,
    qwen4_hc_inject_combine,
)

from . import conftest as cfg

pytestmark = [
    pytest.mark.hyperconnection,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


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


def _inputs(
    rows: int, dtype: torch.dtype, *, shared_weight: bool
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(701 + rows + int(shared_weight))
    hc_count = 4
    hidden_size = 2560
    residual = torch.randn(rows, hc_count * hidden_size, device="cuda", dtype=dtype)
    block = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
    # The injection view is a split from the packed 320+HC projection.  It is
    # contiguous in its last dimension but keeps a widened row stride.
    packed = torch.randn(rows, 320 + hc_count, device="cuda", dtype=dtype)
    injection = packed[:, 320:]
    assert injection.stride() == (324, 1)
    weight = (
        torch.randn(
            hidden_size if shared_weight else hc_count * hidden_size,
            device="cuda",
            dtype=dtype,
        )
        * 0.01
    )
    return residual, block, injection, weight


@pytest.mark.parametrize("rows", [1, 3] if cfg.QUICK_MODE else [1, 3, 17])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shared_weight", [True, False])
def test_combine_norm_rounding_and_packed_stride(
    rows: int, shared_weight: bool, dtype: torch.dtype
) -> None:

    residual, block, injection, weight = _inputs(
        rows, dtype, shared_weight=shared_weight
    )
    assert can_use_hc_combine_norm_triton(injection, block, residual, weight)
    eps = 1.0e-6
    out, normed = qwen4_hc_combine_norm(residual, block, injection, weight, eps, 4)
    ref_out, ref_normed = _reference_combine_norm(
        residual, block, injection, weight, eps, 4
    )
    if weight.numel() == residual.shape[-1]:
        baseline_out = qwen4_hc_inject_combine(injection, block, residual, 4)
        baseline_normed = qwen4_grouped_gemma_rmsnorm(baseline_out, weight, 4, eps)
        torch.testing.assert_close(baseline_normed, ref_normed, atol=4e-2, rtol=3e-2)
    torch.testing.assert_close(out, ref_out, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(normed, ref_normed, atol=4.0e-2, rtol=3.0e-2)
    assert out.dtype is dtype
    assert normed.dtype is dtype
    assert out.is_contiguous()
    assert normed.is_contiguous()


def test_combine_norm_cuda_graph_replay_is_deterministic() -> None:
    residual, block, injection, weight = _inputs(5, torch.bfloat16, shared_weight=False)
    eps = 1.0e-6

    # Eager warmup compiles/lazily allocates before capture.
    eager = qwen4_hc_combine_norm(residual, block, injection, weight, eps, 4)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_normed = qwen4_hc_combine_norm(
            residual, block, injection, weight, eps, 4
        )
    first_inputs = [x.clone() for x in (residual, block, injection, weight)]
    first_ref = _reference_combine_norm(*first_inputs, eps, 4)
    torch.testing.assert_close(eager[0], first_ref[0], atol=2.0e-2, rtol=2.0e-2)

    for iteration in range(10):
        residual.copy_(torch.randn_like(residual))
        block.copy_(torch.randn_like(block))
        injection.copy_(torch.randn_like(injection))
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference_combine_norm(residual, block, injection, weight, eps, 4)
        torch.testing.assert_close(
            graph_out, ref[0], atol=2.0e-2, rtol=2.0e-2, msg=f"output iter {iteration}"
        )
        torch.testing.assert_close(
            graph_normed,
            ref[1],
            atol=4.0e-2,
            rtol=3.0e-2,
            msg=f"norm iter {iteration}",
        )

    residual.copy_(torch.full_like(residual, 0.125))
    block.copy_(torch.full_like(block, -0.25))
    injection.copy_(torch.full_like(injection, 0.5))
    graph.replay()
    torch.cuda.synchronize()
    repeat = graph_out.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, repeat)
