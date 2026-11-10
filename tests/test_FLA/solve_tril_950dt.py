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
import torch

from baseline import chunk_gated_delta_rule_fwd_intra_npu as workload

pytestmark = [
    pytest.mark.solve_tril_950dt,
    pytest.mark.skipif(
        not (hasattr(torch, "npu") and torch.npu.is_available()),
        reason="Ascend NPU required",
    ),
]


def _subject():
    from flaggems_vllm.runtime.backend._ascend.FLA import solve_tril_950dt

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


@pytest.mark.parametrize("case", workload.CASE_NAMES)
def test_is_really_the_inverse(case):
    """``(I + M) Ai = I`` per tile, independent of the reference.

    The comparison above would pass if both sides agreed on a wrong matrix.
    This checks the defining property instead, in float64 on the host.
    """
    kwargs = _kwargs(case)
    m = kwargs["A"]
    ai = _subject()(**kwargs)
    bt = m.shape[-1]
    eye = torch.eye(bt, dtype=torch.float64)
    worst = 0.0
    t = m.shape[1]
    for start in range(0, t - t % bt, bt):
        m0 = m[0, start : start + bt, 0, :].cpu().double()
        a0 = ai[0, start : start + bt, 0, :].cpu().double()
        worst = max(worst, ((eye + m0) @ a0 - eye).abs().max().item())
    # bf16/fp16 storage of Ai bounds how close this can get; 5e-2 is loose
    # enough for fp16 tiles and still rejects a wrong matrix by orders of
    # magnitude (a truncated series leaves ~2e-1).
    assert worst < 5e-2, f"{case}: ||(I+M)Ai - I|| = {worst:.3e}"


@pytest.mark.parametrize("case", workload.CASE_NAMES)
def test_diagonal_is_one(case):
    """The inverse of a unit lower triangular matrix is unit lower triangular.

    Pins the output convention: this returns the *full* inverse, not
    ``inverse - I``. Getting that wrong shifts every downstream w/u by one
    copy of the right-hand side.
    """
    kwargs = _kwargs(case)
    ai = _subject()(**kwargs)
    bt = ai.shape[-1]
    t = ai.shape[1]
    rows = torch.arange(bt)
    for start in range(0, t - t % bt, bt):
        tile = ai[0, start : start + bt, 0, :].float().cpu()
        diag = tile[rows, rows]
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-2), (
            f"{case}: diagonal is {diag[:4].tolist()}..., expected all ones"
        )
        assert tile.triu(1).abs().max().item() == 0.0, (
            f"{case}: upper triangle is nonzero"
        )


def test_all_three_kernels_are_exercised():
    """bt16/bt32/bt64 must really reach three different kernels.

    ``solve_tril`` dispatches on ``A.shape[-1]``, so if these cases ever
    stopped differing in chunk_size, two of the three inverse kernels would
    silently go untested.
    """
    seen = set()
    for case in ("bt16", "bt32", "bt64"):
        kwargs = _kwargs(case)
        seen.add(kwargs["A"].shape[-1])
        _subject()(**kwargs)
    assert seen == {16, 32, 64}, f"expected BT 16/32/64, got {sorted(seen)}"
