# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Accuracy: recompute_w_u_fwd_950dt vs the FLA operator it replaces.

    pytest -v -s tests/test_FLA/recompute_w_u_fwd_950dt.py \
        -m recompute_w_u_fwd_950dt

Stage 3 of ``chunk_gated_delta_rule_fwd_intra_npu``: applies the inverted WY
matrix, ``w = Ai @ (beta * 2^g * k)`` and ``u = Ai @ (beta * v)``. Reference
is ``fla.ops.gated_delta_rule.wy_fast.recompute_w_u_fwd``.

``A`` is produced by the *reference* stages 1 and 2, so both sides consume
the same matrix.
"""

import pytest
import torch

from baseline import chunk_gated_delta_rule_fwd_intra_npu as workload

pytestmark = [
    pytest.mark.recompute_w_u_fwd_950dt,
    pytest.mark.skipif(
        not (hasattr(torch, "npu") and torch.npu.is_available()),
        reason="Ascend NPU required",
    ),
]


def _subject():
    from flaggems_vllm.runtime.backend._ascend.FLA import recompute_w_u_fwd_950dt

    return recompute_w_u_fwd_950dt


def _kwargs(case):
    kwargs = workload.build_recompute_w_u(case)
    if kwargs is None:
        pytest.skip(f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}")
    return kwargs


def _reference():
    native = workload.native_recompute_w_u()
    if native is None:
        pytest.skip(f"reference unavailable: {workload.NATIVE_UNAVAILABLE_REASON}")
    return native


@pytest.mark.parametrize("case", workload.CASE_NAMES)
def test_accuracy(case):
    kwargs = _kwargs(case)
    w_ref, u_ref = _reference()(**kwargs)
    w_got, u_got = _subject()(**kwargs)
    tol = workload.TOL[workload.case_dtype(case)]
    for name, got, want in (("w", w_got, w_ref), ("u", u_got, u_ref)):
        assert got.shape == want.shape, (
            f"{name}: {tuple(got.shape)} != {tuple(want.shape)}"
        )
        assert got.dtype == want.dtype, f"{name}: {got.dtype} != {want.dtype}"
        rel = workload.rel_error(got, want)
        assert rel < tol, f"{case}/{name}: relative error {rel:.3e} exceeds {tol:.1e}"


@pytest.mark.parametrize("case", ["bt64", "hv8", "varlen"])
def test_matches_a_host_reference(case):
    """Check w/u against the formula directly, not only against the FLA op.

    w_i = sum_j Ai[i,j] * beta_j * 2^g_j * k_j and u_i = sum_j Ai[i,j] *
    beta_j * v_j, evaluated in float64 on the host. This catches a shared
    misreading of the gate convention -- using exp instead of exp2, or
    treating g as a raw log decay rather than a per-chunk cumulative sum --
    which a comparison against the reference alone cannot see.
    """
    kwargs = _kwargs(case)
    w_got, u_got = _subject()(**kwargs)
    ai = kwargs["A"].float().cpu().double()
    k = kwargs["k"].float().cpu().double()
    v = kwargs["v"].float().cpu().double()
    beta = kwargs["beta"].float().cpu().double()
    g = kwargs["g"].float().cpu().double()
    bt = ai.shape[-1]
    b, t, hv = beta.shape
    h = k.shape[2]
    group = hv // h
    tol = workload.TOL[workload.case_dtype(case)]

    for i_b in range(b):
        for i_hv in range(hv):
            i_h = i_hv // group
            for start in range(0, t - t % bt, bt):
                sl = slice(start, start + bt)
                a_tile = ai[i_b, sl, i_hv, :]
                rhs_w = (beta[i_b, sl, i_hv] * torch.pow(2.0, g[i_b, sl, i_hv]))[
                    :, None
                ] * k[i_b, sl, i_h]
                rhs_u = beta[i_b, sl, i_hv][:, None] * v[i_b, sl, i_hv]
                exp_w = a_tile @ rhs_w
                exp_u = a_tile @ rhs_u
                got_w = w_got[i_b, sl, i_hv].float().cpu().double()
                got_u = u_got[i_b, sl, i_hv].float().cpu().double()
                for name, got, want in (("w", got_w, exp_w), ("u", got_u, exp_u)):
                    scale = want.abs().max().clamp(min=1e-6)
                    rel = ((got - want).abs().max() / scale).item()
                    assert rel < tol, (
                        f"{case}/{name} b{i_b} h{i_hv} chunk@{start}: "
                        f"relative error {rel:.3e} exceeds {tol:.1e}"
                    )


def test_without_gate():
    """g=None must be accepted: w then carries no decay factor."""
    kwargs = _kwargs("bt64")
    kwargs["g"] = None
    w_ref, u_ref = _reference()(**kwargs)
    w_got, u_got = _subject()(**kwargs)
    tol = workload.TOL[workload.case_dtype("bt64")]
    for name, got, want in (("w", w_got, w_ref), ("u", u_got, u_ref)):
        rel = workload.rel_error(got, want)
        assert rel < tol, f"no-gate/{name}: relative error {rel:.3e} exceeds {tol:.1e}"
