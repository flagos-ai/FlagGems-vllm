# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Accuracy: solve_tril_950dt vs the FLA operator it replaces.

    pytest -v -s tests/test_FLA/solve_tril_950dt.py -m solve_tril_950dt

Stage 2 of ``chunk_gated_delta_rule_fwd_intra_npu``: inverts ``I + M`` per
chunk tile. Reference is ``fla.ops.utils.solve_tril.solve_tril``, which on
NPU dispatches to ``solve_tril_npu`` and picks one of three kernels by
``A.shape[-1]`` alone -- hence the bt16/bt32/bt64 cases, which differ only
in chunk_size and are the only way to reach all three.

``A`` is produced by the *reference* stage 1, so both sides invert the same
matrix and an upstream difference cannot be mistaken for an inverse defect.
"""

import pytest

from baseline import chunk_gated_delta_rule_fwd_intra_npu as workload

pytestmark = [
    pytest.mark.solve_tril_950dt,
    pytest.mark.skipif(
        not workload.IS_SUPPORTED_SOC,
        reason=workload.unsupported_soc_reason() or "",
    ),
]


def _subject():
    from flaggems_vllm.runtime.backend._ascend.A5 import solve_tril_950dt

    return solve_tril_950dt


def _kwargs(case):
    kwargs = workload.build_solve_tril(case)
    if kwargs is None:
        pytest.skip(f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}")
    return kwargs


def _reference():
    native = workload.native_solve_tril()
    if native is None:
        pytest.skip(f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}")
    return native


@pytest.mark.parametrize("case", workload.CASE_NAMES)
def test_accuracy(case):
    kwargs = _kwargs(case)
    want = _reference()(**kwargs)
    got = _subject()(**kwargs)
    assert got.shape == want.shape, f"{tuple(got.shape)} != {tuple(want.shape)}"
    assert got.dtype == want.dtype, f"{got.dtype} != {want.dtype}"
    tol = workload.TOL[workload.case_dtype(case)]
    rel = workload.rel_error(got, want)
    assert rel < tol, f"{case}: relative error {rel:.3e} exceeds {tol:.1e}"
