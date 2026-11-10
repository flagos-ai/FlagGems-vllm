# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Accuracy test: fused (single-kernel) chunk_gated_delta_rule_fwd_intra_npu.

    pytest -v -s tests/test_FLA/chunk_gated_delta_rule_fwd_intra_npu_fused.py \
        -m chunk_gated_delta_rule_fwd_intra_npu_fused

Same cases, inputs and tolerances as the unfused variant -- both come from
``baseline/chunk_gated_delta_rule_fwd_intra_npu.py``.
"""

import pytest
import torch

from baseline import chunk_gated_delta_rule_fwd_intra_npu as workload

pytestmark = [
    pytest.mark.chunk_gated_delta_rule_fwd_intra_npu_fused,
    pytest.mark.skipif(
        not (hasattr(torch, "npu") and torch.npu.is_available()),
        reason="Ascend NPU required",
    ),
]


def _subject():
    from flaggems_vllm.runtime.backend._ascend.FLA import (
        chunk_gated_delta_rule_fwd_intra_npu_fused,
    )

    return chunk_gated_delta_rule_fwd_intra_npu_fused


def _reference():
    native = workload.native()
    if native is None:
        pytest.skip(
            f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}"
        )
    return native


@pytest.mark.parametrize("case", workload.CASE_NAMES)
def test_accuracy(case):
    """The fused variant must match the reference triple (w, u, A)."""
    data = workload.build(case)
    want = _reference()(**data)
    got = _subject()(**data)
    workload.assert_close(case, got, want)


@pytest.mark.parametrize("case", ["bt64", "varlen"])
def test_a_is_really_the_inverse(case):
    """A must satisfy (I + A_kkt) A = I.

    The fused kernel never materializes A_kkt in GM, so this is the only
    check that its in-UB inverse is the right matrix rather than merely
    consistent with its own w/u.
    """
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

    data = workload.build(case)
    bt = data["chunk_size"]
    m = chunk_scaled_dot_kkt_fwd(
        k=data["k"], g=data["g"], beta=data["beta"],
        cu_seqlens=data["cu_seqlens"], chunk_size=bt, output_dtype=torch.float32,
    )
    _w, _u, a = _subject()(**data)
    eye = torch.eye(bt, dtype=torch.float64)
    m0 = m[0, :bt, 0, :].cpu().double()
    a0 = a[0, :bt, 0, :].cpu().double()
    residual = ((eye + m0) @ a0 - eye).abs().max().item()
    assert residual < 5e-2, f"{case}: ||(I+M)A - I|| = {residual:.3e}"


def test_matches_unfused_variant():
    """Fused and unfused must agree with each other, not just with the reference.

    They share no kernel code, so agreement here is independent evidence that
    the fused doubling-based inverse and the three-stage path compute the same
    thing.
    """
    from flaggems_vllm.runtime.backend._ascend.FLA import (
        chunk_gated_delta_rule_fwd_intra_npu_unfused,
    )

    for case in ("bt64", "d128", "varlen"):
        data = workload.build(case)
        fused = _subject()(**data)
        unfused = chunk_gated_delta_rule_fwd_intra_npu_unfused(**data)
        workload.assert_close(case, fused, unfused)


def test_unsupported_paths_raise():
    """Unsupported inputs raise instead of degrading silently."""
    subject = _subject()
    b, t, h, hv, d = 1, 64, 2, 2, 64
    zeros = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device="npu")  # noqa: E731
    k, v = zeros(b, t, h, d), zeros(b, t, hv, d)
    beta = zeros(b, t, hv)

    with pytest.raises(ValueError):
        subject(k=k, v=v, beta=beta, chunk_size=48)
    with pytest.raises(NotImplementedError):
        subject(k=k, v=v, beta=None)
    with pytest.raises(NotImplementedError):
        big = zeros(b, t, h, 256)
        subject(k=big, v=zeros(b, t, hv, 256), beta=beta)
    with pytest.raises(ValueError):
        subject(k=k.squeeze(0), v=v, beta=beta)
