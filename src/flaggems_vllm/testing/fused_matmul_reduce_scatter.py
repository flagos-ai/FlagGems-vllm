# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Test-only FP32 reference and conditioning-aware low-precision validation."""

import torch
import torch.distributed as dist


def make_exact_case(group, device, dtype, reduce_op):
    """Encode rank/row/column identity with exactly representable results.

    For 2/4/8 ranks, all products and hierarchical subtotals are representable
    in both FP16 and BF16. This case must pass with zero numerical tolerance.
    """
    if group.size() not in (2, 4, 8):
        raise ValueError("the exact encoding supports 2, 4 or 8 ranks")
    rows = 256
    row_ids = torch.arange(rows * group.size(), device=device)
    codes = ((row_ids // rows) * 8 + row_ids % 8).float()
    scales = torch.tensor([0.0625, 0.125, 0.25, 0.5], device=device).repeat(64)
    a = torch.zeros((rows * group.size(), 64), device=device, dtype=dtype)
    b = torch.zeros((64, 256), device=device, dtype=dtype)
    a[:, 0] = codes + group.rank() + 1
    b[0] = scales
    rank_sum = group.size() * (group.size() + 1) // 2
    expected = (group.size() * codes + rank_sum)[:, None] * scales[None, :]
    if reduce_op == "avg":
        expected /= group.size()
    start = rows * group.rank()
    return a, b, expected[start : start + rows].contiguous()


def roundoff_tolerance(expected, sum_abs_partials, dtype):
    """Account for low-precision writes, including cancellation between ranks.

    GEMM output, the node subtotal, and the final output can each be rounded to
    FP16/BF16. For round-to-nearest unit roundoff u=eps/2, gamma_3 bounds these
    three storage roundings against sum(abs(partial)), rather than abs(sum).
    The existing base atol/rtol also cover FP32 arithmetic/reference differences.
    This is a storage-roundoff budget, not a full GEMM forward-error proof.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("reference validation requires FP16 or BF16 inputs")
    rtol, atol = (0.04, 0.005) if dtype == torch.bfloat16 else (0.01, 0.001)
    unit_roundoff = torch.finfo(dtype).eps / 2
    gamma = 3 * unit_roundoff / (1 - 3 * unit_roundoff)
    return atol + rtol * expected.abs() + gamma * sum_abs_partials


def make_reference(a, b, reduce_op, group):
    """Evaluate the mathematical result and partial-result scale in FP32.

    The low-precision Torch GEMM+collective remains a performance baseline;
    it is not the accuracy oracle. No implementation scratch buffers are read.
    """
    if reduce_op not in ("sum", "avg"):
        raise ValueError("reference reduction must be sum or avg")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        partial = a.float() @ b.float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    shape = (a.shape[0] // group.size(), b.shape[1])
    expected = torch.empty(shape, device=a.device, dtype=torch.float32)
    sum_abs_partials = torch.empty_like(expected)
    dist.reduce_scatter_tensor(expected, partial, group=group)
    dist.reduce_scatter_tensor(sum_abs_partials, partial.abs(), group=group)
    if reduce_op == "avg":
        expected.div_(group.size())
        sum_abs_partials.div_(group.size())
    return expected, roundoff_tolerance(expected, sum_abs_partials, a.dtype)


def error_metrics(actual, expected, tolerance):
    if actual.shape != expected.shape or tolerance.shape != expected.shape:
        raise AssertionError("output/reference/tolerance shapes must match")
    delta = (actual.float() - expected).abs()
    bad = (
        ~torch.isfinite(actual)
        | ~torch.isfinite(expected)
        | ~torch.isfinite(tolerance)
        | (delta > tolerance)
    )
    return {
        "max_abs": delta.max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "outside_budget": bad.sum().item(),
        "elements": delta.numel(),
        "max_budget": tolerance.max().item(),
    }


def assert_close(actual, expected, tolerance, group, *, dtype):
    """Fail on every group rank if one rank exceeds the FP32 error budget."""
    message = None
    try:
        if actual.dtype != dtype or actual.device != expected.device:
            raise AssertionError(
                "output dtype/device does not match the operator contract"
            )
        metrics = error_metrics(actual, expected, tolerance)
        if metrics["outside_budget"]:
            message = f"FP32 reference check failed: {metrics}"
    except AssertionError as error:
        message = str(error)
    failed = torch.tensor(int(message is not None), device=expected.device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
    if failed.item():
        raise AssertionError(message or "a peer rank failed FP32 reference validation")
