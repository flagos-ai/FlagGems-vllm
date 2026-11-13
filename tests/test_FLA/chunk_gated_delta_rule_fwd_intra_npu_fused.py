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
        not workload.IS_SUPPORTED_SOC,
        reason=workload.unsupported_soc_reason() or "",
    ),
]


def _subject():
    from flaggems_vllm.runtime.backend._ascend.A5 import (
        chunk_gated_delta_rule_fwd_intra_npu_fused,
    )

    return chunk_gated_delta_rule_fwd_intra_npu_fused


def _reference():
    native = workload.native()
    if native is None:
        pytest.skip(f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}")
    return native


@pytest.mark.parametrize("case", workload.CASE_NAMES)
def test_accuracy(case):
    """The fused variant must match the reference triple (w, u, A)."""
    data = workload.build(case)
    want = _reference()(**data)
    got = _subject()(**data)
    workload.assert_close(case, got, want)


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
