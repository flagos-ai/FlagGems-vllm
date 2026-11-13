# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Accuracy: chunk_scaled_dot_kkt_fwd_950dt vs the FLA operator it replaces.

    pytest -v -s tests/test_FLA/chunk_scaled_dot_kkt_fwd_950dt.py \
        -m chunk_scaled_dot_kkt_fwd_950dt

Stage 1 of ``chunk_gated_delta_rule_fwd_intra_npu``: builds
``M = tril(beta * K K^T * 2^(g_i - g_j), -1)``. Reference is
``fla.ops.common.chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd``.

Cases and inputs come from ``baseline/chunk_gated_delta_rule_fwd_intra_npu.py``,
shared with the other five per-operator files and with the pipeline tests.
"""

import pytest
import torch

from baseline import chunk_gated_delta_rule_fwd_intra_npu as workload

pytestmark = [
    pytest.mark.chunk_scaled_dot_kkt_fwd_950dt,
    pytest.mark.skipif(
        not (hasattr(torch, "npu") and torch.npu.is_available()),
        reason="Ascend NPU required",
    ),
]


def _subject():
    from flaggems_vllm.runtime.backend._ascend.A5 import chunk_scaled_dot_kkt_fwd_950dt

    return chunk_scaled_dot_kkt_fwd_950dt


def _reference():
    native = workload.native_kkt()
    if native is None:
        pytest.skip(f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}")
    return native


@pytest.mark.parametrize("case", workload.CASE_NAMES)
def test_accuracy(case):
    kwargs = workload.build_kkt(case)
    want = _reference()(**kwargs)
    got = _subject()(**kwargs)
    assert got.shape == want.shape, f"{tuple(got.shape)} != {tuple(want.shape)}"
    assert got.dtype == want.dtype, f"{got.dtype} != {want.dtype}"
    tol = workload.TOL[workload.case_dtype(case)]
    rel = workload.rel_error(got, want)
    assert rel < tol, f"{case}: relative error {rel:.3e} exceeds {tol:.1e}"


def test_without_gate():
    """g=None must be accepted and match the reference (no decay factor)."""
    kwargs = workload.build_kkt("bt64")
    kwargs["g"] = None
    want = _reference()(**kwargs)
    got = _subject()(**kwargs)
    tol = workload.TOL[workload.case_dtype("bt64")]
    rel = workload.rel_error(got, want)
    assert rel < tol, f"no-gate: relative error {rel:.3e} exceeds {tol:.1e}"
