# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Workload definition for ``chunk_gated_delta_rule_fwd_intra_npu``.

Single source of the test cases, inputs and reference op, shared by all four
accuracy/benchmark files so the unfused and fused variants are judged on
identical work:

- ``tests/test_FLA/chunk_gated_delta_rule_fwd_intra_npu.py``
- ``tests/test_FLA/chunk_gated_delta_rule_fwd_intra_npu_fused.py``
- ``benchmark/chunk_gated_delta_rule_fwd_intra_npu.py``
- ``benchmark/chunk_gated_delta_rule_fwd_intra_npu_fused.py``

The reference is flash-linear-attention's
``fla.ops.gated_delta_rule.backends.triton_ascend.chunk_fwd
.chunk_gated_delta_rule_fwd_intra_npu``, which chains three operators:
``chunk_scaled_dot_kkt_fwd`` -> ``solve_tril`` -> ``recompute_w_u_fwd``.

Input construction follows FLA's own kernel test
(``tests/ops/test_gdn_kernels.py::test_chunk_kkt_solve``): k is L2
normalized, beta goes through sigmoid, g is ``randn * 0.1``. Two conventions
must match the reference exactly or the numbers look plausible but are wrong
-- ``g`` is a per-chunk cumulative sum, and the decay uses ``exp2``, not
``exp``.
"""

import torch

OP_NAME = "chunk_gated_delta_rule_fwd_intra_npu"

DTYPES = [torch.bfloat16, torch.float16]

NATIVE_UNAVAILABLE_REASON = None

# name -> (B, T, H, HV, D, dtype, chunk_size, seqs or None)
#
# ``chunk_size`` is a deliberate axis, not a free parameter: ``solve_tril``
# dispatches purely on ``A.shape[-1]``, so it alone selects between the
# 16x16, 32x32 and 64x64 inverse kernels. Testing only 64 leaves the other
# two kernels never compiled and never measured.
CASES = {
    "bt64": (2, 128, 2, 2, 64, torch.bfloat16, 64, None),
    "bt32": (2, 128, 2, 2, 64, torch.bfloat16, 32, None),
    "bt16": (2, 128, 2, 2, 64, torch.bfloat16, 16, None),
    "hv8": (2, 256, 2, 8, 64, torch.bfloat16, 64, None),
    "d128": (2, 256, 4, 4, 128, torch.bfloat16, 64, None),
    "large": (2, 1024, 4, 8, 128, torch.bfloat16, 64, None),
    "fp16": (1, 256, 4, 4, 32, torch.float16, 64, None),
    "varlen": (1, 320, 2, 2, 64, torch.bfloat16, 64, (64, 128, 128)),
}

CASE_NAMES = list(CASES)


def native():
    """The reference op this workload is measured against."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        from fla.ops.gated_delta_rule.backends.triton_ascend.chunk_fwd import (
            chunk_gated_delta_rule_fwd_intra_npu,
        )
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    return chunk_gated_delta_rule_fwd_intra_npu


def grid():
    return [{"case": name} for name in CASE_NAMES]


def build(case, device="npu"):
    """Materialize one case. Returns the kwargs both sides are called with."""
    b, t, h, hv, d, dtype, bt, seqs = CASES[case]
    torch.manual_seed(42)
    k = torch.nn.functional.normalize(
        torch.randn(b, t, h, d, dtype=dtype, device=device), p=2, dim=-1
    ).contiguous()
    v = torch.randn(b, t, hv, d, dtype=dtype, device=device).contiguous()
    beta = torch.rand(b, t, hv, dtype=dtype, device=device).sigmoid().contiguous()
    g = (torch.randn(b, t, hv, dtype=torch.float32, device=device) * 0.1).contiguous()
    cu_seqlens = None
    if seqs is not None:
        offsets = [0]
        for length in seqs:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
    return dict(k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=bt)


def build_inputs(binding, dtype, device):
    """Benchmark hook. ``dtype`` is fixed per case, so it is ignored here.

    Each case pins its own dtype (``fp16`` is a separate case), and changing
    it would make the case something other than what it was chosen to cover.
    """
    del dtype
    return (), build(binding["case"], device)


def key_shape(binding):
    return binding["case"]


def case_dtype(case):
    return CASES[case][5]


def config(binding, dtype):
    del dtype
    b, t, h, hv, d, case_dt, bt, seqs = CASES[binding["case"]]
    return dict(
        case=binding["case"],
        B=b,
        T=t,
        H=h,
        HV=hv,
        D=d,
        dtype=str(case_dt),
        chunk_size=bt,
        seqs=list(seqs) if seqs else None,
    )


def unsupported_reason(binding, dtype):
    del binding, dtype
    return None


# Relative tolerance per dtype, applied against each tensor's own magnitude.
# u reaches ~3 in these cases while A is O(1), so an absolute bound would
# either pass trivially or fail on rounding.
TOL = {
    torch.bfloat16: 2e-2,
    torch.float16: 4e-3,
    torch.float32: 1e-5,
}


def assert_close(case, got, want):
    """Compare the (w, u, A) triple against the reference."""
    dtype = case_dtype(case)
    tol = TOL[dtype]
    for name, g, w in zip(("w", "u", "A"), got, want):
        assert g.shape == w.shape, f"{name}: shape {tuple(g.shape)} != {tuple(w.shape)}"
        gf, wf = g.float(), w.float()
        scale = wf.abs().max().clamp(min=1e-6)
        rel = ((gf - wf).abs().max() / scale).item()
        assert rel < tol, f"{case}/{name}: relative error {rel:.3e} exceeds {tol:.1e}"


def rel_error(got, want):
    """Relative error against ``want``'s own magnitude."""
    gf, wf = got.float(), want.float()
    return ((gf - wf).abs().max() / wf.abs().max().clamp(min=1e-6)).item()


# --------------------------------------------------------------------------
# Per-operator workloads
#
# The three operators inside chunk_gated_delta_rule_fwd_intra_npu are
# compared one at a time. Each stage is fed the *reference* pipeline's input
# for that stage, never the previous stage's own output: otherwise an error
# in stage 1 propagates and shows up as a stage 2 defect, and a stage can
# also be flattered by consuming its own (differently rounded) upstream.
# --------------------------------------------------------------------------


def native_kkt():
    """`chunk_scaled_dot_kkt_fwd` -- stage 1 of the reference pipeline."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    return chunk_scaled_dot_kkt_fwd


def native_solve_tril():
    """`solve_tril` -- stage 2. On NPU this dispatches to solve_tril_npu."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        from fla.ops.utils import solve_tril
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    return solve_tril


def native_recompute_w_u():
    """`recompute_w_u_fwd` -- stage 3 of the reference pipeline."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    return recompute_w_u_fwd


def build_kkt(case, device="npu"):
    """Stage-1 kwargs: straight from the case inputs."""
    data = build(case, device)
    return dict(
        k=data["k"],
        g=data["g"],
        beta=data["beta"],
        cu_seqlens=data["cu_seqlens"],
        chunk_size=data["chunk_size"],
        output_dtype=torch.float32,
    )


def build_solve_tril(case, device="npu"):
    """Stage-2 kwargs: A comes from the *reference* stage 1."""
    data = build(case, device)
    kkt = native_kkt()
    if kkt is None:
        return None
    a = kkt(
        k=data["k"],
        g=data["g"],
        beta=data["beta"],
        cu_seqlens=data["cu_seqlens"],
        chunk_size=data["chunk_size"],
        output_dtype=torch.float32,
    )
    return dict(
        A=a,
        cu_seqlens=data["cu_seqlens"],
        output_dtype=data["k"].dtype,
    )


def build_recompute_w_u(case, device="npu"):
    """Stage-3 kwargs: A comes from the *reference* stages 1 and 2."""
    data = build(case, device)
    kkt, solve = native_kkt(), native_solve_tril()
    if kkt is None or solve is None:
        return None
    a = kkt(
        k=data["k"],
        g=data["g"],
        beta=data["beta"],
        cu_seqlens=data["cu_seqlens"],
        chunk_size=data["chunk_size"],
        output_dtype=torch.float32,
    )
    ai = solve(A=a, cu_seqlens=data["cu_seqlens"], output_dtype=data["k"].dtype)
    return dict(
        k=data["k"],
        v=data["v"],
        beta=data["beta"],
        A=ai,
        g=data["g"],
        cu_seqlens=data["cu_seqlens"],
    )


# Benchmark hooks, one set per operator. Shapes are the shared case names so
# all six files sweep the same workloads.
def build_inputs_kkt(binding, dtype, device):
    del dtype
    return (), build_kkt(binding["case"], device)


def build_inputs_solve_tril(binding, dtype, device):
    del dtype
    return (), build_solve_tril(binding["case"], device)


def build_inputs_recompute_w_u(binding, dtype, device):
    del dtype
    return (), build_recompute_w_u(binding["case"], device)
