# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent materialization + native FGV selection reference (tests only)."""
import torch
import triton
import triton.language as tl

from flaggems_vllm import grouped_topk
from flaggems_vllm.utils import tl_extra_shim

_exp2 = tl_extra_shim.exp2


@triton.jit
def _materialize_sigmoid(
    X, Y, E: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * STRIDE + col, col < E, 0).to(tl.float32)
    probability = 1.0 / (1.0 + _exp2(-value * 1.4426950408889634))
    tl.store(Y + row * E + col, probability.to(Y.dtype.element_ty), col < E)


def reference(scores, bias):
    processed = torch.empty(scores.shape, dtype=scores.dtype, device=scores.device)
    _materialize_sigmoid[(scores.shape[0],)](
        scores,
        processed,
        scores.shape[1],
        scores.stride(0),
        triton.next_power_of_2(scores.shape[1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return grouped_topk(
        processed,
        1,
        1,
        16,
        True,
        1.0,
        bias.to(scores.dtype),
        0,
    )


def candidate(scores, bias):
    return grouped_topk(
        scores,
        1,
        1,
        16,
        True,
        1.0,
        bias,
        1,
        sigmoid_mode="fp32_then_input",
    )


def assert_exact(actual, expected):
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[0].view(torch.int32), expected[0].view(torch.int32))
