"""CPU-only check: measure max||L||_2 under our real KDA input distribution,
to see whether the Neumann series for (I+L)^-1 in _ascend_kda_fwd_intra_kernel
can be truncated below its current order (chains to L^15 via 4 dots: L2, L4,
L8 doublings) -- mirrors the method in cannbot-knowledge's kda_constraints.md
L1.2, but measured on *our* kernel's L construction and *our* input
distribution/environment (Ascend950DT / Triton-Ascend 3.6.0), not borrowed
from that card's Ascend910B2C numbers.

L (our Akk, strictly lower triangular) is built in
src/flaggems_vllm/runtime/backend/_ascend/FLA/chunk_kda.py as:
    Akk[i,j] = beta_i * (k_i . k_j) * 2**(gc_i - gc_j)   for i > j, else 0
where gc is the safe-gate cumsum (same pipeline as
tools/_kda_bt64_subblock_math_check.py), beta = sigmoid(raw_beta),
raw_beta ~ Uniform(0,1) (matches tools/_kda_shape_bench.kda_inputs).

This script reproduces that construction in fp64 (no bf16 rounding, so it
measures the *mathematical* ||L||_2 of the algorithm, not kernel rounding
noise) for BT in {16, 32, 64} and reports:
  - max/mean/p99 spectral norm of L over many random chunks
  - the Neumann remainder bound ||L||^(r+1) / (1-||L||) for r in {1,2,3,4}
  - whether the current r=4 scheme (chained to L^15) is over-provisioned
"""

import torch

torch.manual_seed(0)

K = 128
RCP_LN2 = 1.4426950216
LOWER_BOUND = -5.0
N_TRIALS = 500


def make_chunk(bt: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    raw_g = -torch.rand(bt, K, dtype=torch.float64, generator=g) * 0.05
    a_log = torch.randn(1, dtype=torch.float64, generator=g) * 0.1
    dt_bias = torch.randn(K, dtype=torch.float64, generator=g) * 0.1
    k = torch.randn(bt, K, dtype=torch.float64, generator=g)
    raw_beta = torch.rand(bt, dtype=torch.float64, generator=g)  # matches kda_inputs

    b_A = torch.exp2(a_log * RCP_LN2)
    b_g = raw_g + dt_bias[None, :]
    b_g = (LOWER_BOUND * RCP_LN2) * torch.sigmoid(b_A * b_g)
    gc = torch.cumsum(b_g, dim=0)  # [bt, K], non-increasing

    beta = torch.sigmoid(raw_beta)  # [bt]

    k_rstd = 1.0 / torch.sqrt((k * k).sum(dim=1) + 1e-6)
    k_n = k * k_rstd[:, None]

    diff = gc[:, None, :] - gc[None, :, :]  # [bt, bt, K]
    factor = torch.exp2(diff)
    Akk_full = torch.einsum("id,jd,ijd->ij", k_n, k_n, factor)  # (k_i . k_j) term
    Akk_full = Akk_full * beta[:, None]

    strict_lower = torch.tril(torch.ones(bt, bt, dtype=torch.bool), diagonal=-1)
    L = torch.where(strict_lower, Akk_full, torch.zeros_like(Akk_full))
    return L


def spectral_norm(L: torch.Tensor) -> float:
    # ||L||_2 = largest singular value.
    s = torch.linalg.svdvals(L)
    return s.max().item()


def remainder_bound(norm: float, r: int) -> float:
    if norm >= 1.0:
        return float("inf")
    return norm ** (r + 1) / (1.0 - norm)


def main():
    print(f"N_TRIALS={N_TRIALS} per BT, K={K}, lower_bound={LOWER_BOUND}\n")
    for bt in (16, 32, 64):
        norms = []
        for trial in range(N_TRIALS):
            L = make_chunk(bt, seed=trial + 7000)
            norms.append(spectral_norm(L))
        norms_t = torch.tensor(norms)
        mx = norms_t.max().item()
        mean = norms_t.mean().item()
        p99 = torch.quantile(norms_t, 0.99).item()

        print(f"--- BT={bt} ---")
        print(f"  ||L||_2: max={mx:.4e}  mean={mean:.4e}  p99={p99:.4e}")
        for r in (1, 2, 3, 4):
            print(
                f"    r={r}: remainder bound at max||L||_2 = {remainder_bound(mx, r):.4e}"
            )
        # Current kernel scheme is effectively r up to 15 (chains I-L, +L^2..L^3,
        # +L^4..L^7, +L^8..L^15 via the doubling trick) -- always exact if
        # L^BT == 0, i.e. once r+1 >= bt. Report what the *doubling* scheme
        # actually needs: how many dot-stages until the remainder bound drops
        # below bf16 eps (~2^-8 ~ 3.9e-3) at this BT's max norm.
        bf16_eps = 2.0**-8
        order = 1
        stage = 0
        while remainder_bound(mx, order) > bf16_eps and order < bt:
            order = order * 2 + 1  # mimics the I-L, +L^2, +L^4, +L^8 doubling steps
            stage += 1
        print(
            f"  doubling stages needed for remainder < bf16_eps({bf16_eps:.1e}): "
            f"{stage} stage(s) (current kernel always runs 3 stages -> L2,L4,L8)"
        )
        print()


if __name__ == "__main__":
    main()
