# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Accuracy test: unfused (three-kernel) chunk_gated_delta_rule_fwd_intra_npu.

    pytest -v -s tests/test_FLA/chunk_gated_delta_rule_fwd_intra_npu.py \
        -m chunk_gated_delta_rule_fwd_intra_npu

Cases, inputs and tolerances come from
``baseline/chunk_gated_delta_rule_fwd_intra_npu.py``, shared with the fused
variant and with both benchmarks, so all four files judge identical work.
"""

import pytest
import torch

from baseline import chunk_gated_delta_rule_fwd_intra_npu as workload

pytestmark = [
    pytest.mark.chunk_gated_delta_rule_fwd_intra_npu,
    pytest.mark.skipif(
        not (hasattr(torch, "npu") and torch.npu.is_available()),
        reason="Ascend NPU required",
    ),
]


def _subject():
    from flaggems_vllm.runtime.backend._ascend.A5 import (
        chunk_gated_delta_rule_fwd_intra_npu_unfused,
    )

    return chunk_gated_delta_rule_fwd_intra_npu_unfused


def _reference():
    native = workload.native()
    if native is None:
        pytest.skip(
            f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}"
        )
    return native


@pytest.mark.parametrize("case", workload.CASE_NAMES)
def test_accuracy(case):
    """The three-kernel variant must match the reference triple (w, u, A)."""
    data = workload.build(case)
    want = _reference()(**data)
    got = _subject()(**data)
    workload.assert_close(case, got, want)


@pytest.mark.parametrize("case", workload.CASE_NAMES)
def test_stages_match_their_reference_counterparts(case):
    """Each of the three kernels must match the FLA operator it replaces.

    The end-to-end check above can hide a compensating pair of errors across
    stages; this pins each stage down. Every stage is fed the *reference*
    pipeline's input for that stage, so an error cannot propagate forward
    and be mistaken for a downstream bug.
    """
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
    from fla.ops.utils import solve_tril

    from flaggems_vllm.runtime.backend._ascend.A5 import (
        chunk_scaled_dot_kkt_fwd_950dt,
        recompute_w_u_fwd_950dt,
        solve_tril_950dt,
    )

    data = workload.build(case)
    k, v, g, beta = data["k"], data["v"], data["g"], data["beta"]
    cu, bt = data["cu_seqlens"], data["chunk_size"]
    tol = workload.TOL[workload.case_dtype(case)]

    def rel(got, want):
        gf, wf = got.float(), want.float()
        return ((gf - wf).abs().max() / wf.abs().max().clamp(min=1e-6)).item()

    kw = dict(k=k, g=g, beta=beta, cu_seqlens=cu, chunk_size=bt,
              output_dtype=torch.float32)
    a_ref = chunk_scaled_dot_kkt_fwd(**kw)
    a_mine = chunk_scaled_dot_kkt_fwd_950dt(**kw)
    assert rel(a_mine, a_ref) < tol, f"{case}/kkt: {rel(a_mine, a_ref):.3e}"

    kw = dict(A=a_ref, cu_seqlens=cu, output_dtype=k.dtype)
    ai_ref = solve_tril(**kw)
    ai_mine = solve_tril_950dt(**kw)
    assert rel(ai_mine, ai_ref) < tol, f"{case}/solve_tril: {rel(ai_mine, ai_ref):.3e}"

    kw = dict(k=k, v=v, beta=beta, A=ai_ref, g=g, cu_seqlens=cu)
    w_ref, u_ref = recompute_w_u_fwd(**kw)
    w_mine, u_mine = recompute_w_u_fwd_950dt(**kw)
    assert rel(w_mine, w_ref) < tol, f"{case}/w: {rel(w_mine, w_ref):.3e}"
    assert rel(u_mine, u_ref) < tol, f"{case}/u: {rel(u_mine, u_ref):.3e}"


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
    with pytest.raises(ValueError):
        subject(k=k.squeeze(0), v=v, beta=beta)
