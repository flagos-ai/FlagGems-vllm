"""CPU-only numerical check for BT=64 sub-block rebasing (no NPU required).

Verifies, against an fp64 ground truth and using the *actual* KDA safe-gate
pipeline (lower_bound * sigmoid(A_log-scaled gate), RCP_LN2 units, cumsum --
same as `_ascend_kda_fwd_intra_kernel` in
src/flaggems_vllm/runtime/backend/_ascend/FLA/chunk_kda.py), whether splitting
the BT=64 Aqk/Akk construction into two BT/2=32 sub-blocks (each rebased on
its own local midpoint, plus one cross-block scalar correction for the
strictly-below-diagonal block) stays finite in bf16 where the current single
chunk-wide midpoint does not.

Pure math/dtype probe: no kernel code is touched, no NPU is needed.
"""

import torch

torch.manual_seed(0)

BT = 64
HALF = BT // 2
K = 128
RCP_LN2 = 1.4426950216
LOWER_BOUND = -5.0


def to_bf16_f32(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through bf16 the way the kernel's tl.dot operands do."""
    return x.to(torch.bfloat16).to(torch.float32)


def make_case(seed: int):
    """Reproduce the actual safe-gate pipeline in fp64, matching
    _ascend_kda_fwd_intra_kernel: raw_g -> +dt_bias -> lower_bound*sigmoid(A*g)
    -> cumsum. Uses the same magnitudes as tools/_kda_shape_bench.kda_inputs.
    """
    g = torch.Generator().manual_seed(seed)
    raw_g = -torch.rand(BT, K, dtype=torch.float64, generator=g) * 0.05
    a_log = torch.randn(1, dtype=torch.float64, generator=g) * 0.1
    dt_bias = torch.randn(K, dtype=torch.float64, generator=g) * 0.1
    q = torch.randn(BT, K, dtype=torch.float64, generator=g)
    k = torch.randn(BT, K, dtype=torch.float64, generator=g)

    b_A = torch.exp2(a_log * RCP_LN2)
    b_g = raw_g + dt_bias[None, :]
    b_g = (LOWER_BOUND * RCP_LN2) * torch.sigmoid(b_A * b_g)  # <= 0, matches kernel
    gc = torch.cumsum(b_g, dim=0)  # non-increasing, same as kernel's b_gc
    return gc, q, k


def ground_truth_Aqk(gc, q, k):
    """fp64 exact Aqk[i, j] = (q_i . k_j) * 2**(gc_i - gc_j) for i >= j, else 0."""
    diff = gc[:, None, :] - gc[None, :, :]  # [BT, BT, K]
    factor = torch.exp2(diff)
    Aqk = torch.einsum("id,jd,ijd->ij", q, k, factor)
    causal = torch.tril(torch.ones(BT, BT, dtype=torch.bool))
    return torch.where(causal, Aqk, torch.zeros_like(Aqk))


def single_midpoint_bf16(gc, q, k):
    """Current kernel behaviour: one anchor at row BT//2, bf16 dot."""
    anchor = gc[HALF : HALF + 1, :]
    gq = torch.exp2(gc - anchor)
    gk = torch.exp2(-(gc - anchor))
    qg = to_bf16_f32(q * gq)
    kg = to_bf16_f32(k * gk)
    Aqk = qg @ kg.T
    causal = torch.tril(torch.ones(BT, BT, dtype=torch.bool))
    return torch.where(causal, Aqk, torch.zeros_like(Aqk)), gq, gk


def subblock_rebased_bf16(gc, q, k):
    """Proposed: split BT=64 into two HALF=32 sub-blocks, each with its own
    local midpoint anchor. The strictly-below-diagonal cross block (rows in
    sub-block 1, cols in sub-block 0, all i>=j) gets the scalar cross-anchor
    correction exp2(anchor1 - anchor0) folded into the k-side operand before
    the bf16 cast.
    """
    causal_half = torch.tril(torch.ones(HALF, HALF, dtype=torch.bool))
    Aqk = torch.zeros(BT, BT, dtype=torch.float64)

    local_mid = HALF // 2
    gc0, gc1 = gc[:HALF], gc[HALF:]
    anchor0 = gc0[local_mid : local_mid + 1, :]
    anchor1 = gc1[local_mid : local_mid + 1, :]
    q0, k0 = q[:HALF], k[:HALF]
    q1, k1 = q[HALF:], k[HALF:]

    gq0 = torch.exp2(gc0 - anchor0)
    gk0 = torch.exp2(-(gc0 - anchor0))
    qg0 = to_bf16_f32(q0 * gq0)
    kg0 = to_bf16_f32(k0 * gk0)
    A00 = qg0 @ kg0.T
    Aqk[:HALF, :HALF] = torch.where(causal_half, A00, 0.0)

    gq1 = torch.exp2(gc1 - anchor1)
    gk1 = torch.exp2(-(gc1 - anchor1))
    qg1 = to_bf16_f32(q1 * gq1)
    kg1 = to_bf16_f32(k1 * gk1)
    A11 = qg1 @ kg1.T
    Aqk[HALF:, HALF:] = torch.where(causal_half, A11, 0.0)

    # Cross block: all i>=j, no mask needed. Fold exp2(anchor1-anchor0) into
    # the k0-side factor before the bf16 cast so the matmul operands stay
    # plain bf16 tensors (no fp32 correction multiply after the dot).
    cross_corr = torch.exp2(anchor1 - anchor0)  # [1, K]
    kg0_corrected = to_bf16_f32(k0 * gk0 * cross_corr)
    Aqk[HALF:, :HALF] = qg1 @ kg0_corrected.T

    return Aqk, (gq0, gk0, gq1, gk1, cross_corr)


def report(name, got, ref):
    finite = torch.isfinite(got)
    nf = (~finite).sum().item()
    err = (got - ref).abs()
    err_finite = err[finite]
    max_err = err_finite.max().item() if err_finite.numel() else float("nan")
    rel = max_err / (ref.abs().max().item() + 1e-30)
    max_out = got[finite].abs().max().item() if finite.any() else float("nan")
    print(
        f"{name:28s} max_abs_err={max_err:12.4e} rel_err={rel:10.4e} "
        f"nonfinite={nf:4d}/{got.numel():<5d} max|out|={max_out:12.4e}"
    )


def main():
    print(
        f"BT={BT}, HALF={HALF}, K={K}, lower_bound={LOWER_BOUND}, RCP_LN2={RCP_LN2}\n"
    )
    for seed in range(5):
        print(f"--- seed={seed} (real safe-gate pipeline, matches kernel) ---")
        gc, q, k = make_case(seed)
        span_above = (gc.max() - gc[HALF].min()).item()
        span_below = (gc[HALF].max() - gc.min()).item()
        print(
            f"  gc range over chunk: [{gc.min().item():.3f}, {gc.max().item():.3f}] "
            f"(log2 units); single-anchor worst-case factor: "
            f"exp2({span_above:.3f}) / exp2({span_below:.3f})"
        )
        ref = ground_truth_Aqk(gc, q, k)

        got_single, gq_s, gk_s = single_midpoint_bf16(gc, q, k)
        report("single_midpoint(BT=64)", got_single, ref)

        got_sub, _ = subblock_rebased_bf16(gc, q, k)
        report("subblock_rebase(2x32)", got_sub, ref)
        print()


if __name__ == "__main__":
    main()
