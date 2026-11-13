import pytest
import torch

from tests.test_mhc_ops import sinkhorn_normalize_ref

from flaggems_vllm.ops.mhc.mhc_pre_broadcast import (
    mhc_pre_broadcast,
    derive_fn_broadcast,
)


def generate_mhc_pre_broadcast_data(n, hc_mult, hidden_size, device="cuda"):
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    residual = torch.randn(n, hidden_size, device=device).bfloat16()
    fn = torch.randn(hc_mult3, hc_hidden_size, device=device, dtype=torch.float32)
    fn_broadcast = derive_fn_broadcast(fn, hc_mult)
    hc_scale = torch.rand(3, device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    norm_weight = torch.randn(hidden_size, device=device).bfloat16()
    return dict(
        residual=residual,
        fn_broadcast=fn_broadcast,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0,
        sinkhorn_repeat=3,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )


MHC_PRE_BROADCAST_CONFIGS = [(1, 1280), (8, 1280), (128, 5120), (4096, 7168)]


@pytest.mark.mhc_pre_broadcast
@pytest.mark.parametrize(
    "n, hidden_size",
    MHC_PRE_BROADCAST_CONFIGS,
    ids=[f"n{n}_h{h}" for n, h in MHC_PRE_BROADCAST_CONFIGS],
)
def test_mhc_pre_broadcast_vs_ref(n, hidden_size):
    torch.manual_seed(0)
    data = generate_mhc_pre_broadcast_data(n, 4, hidden_size)
    out_triton = mhc_pre_broadcast(**data)
    data_cpu = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in data.items()}
    out_ref = mhc_pre_broadcast_ref(**data_cpu)
    names = ["residual_out", "post_mix", "comb_mix", "layer_input"]
    assert torch.equal(out_triton[0].cpu(), out_ref[0]), "residual_out must be bit-exact"
    for name, a, b in zip(names, out_triton, out_ref):
        a_cpu, b_cpu = a.cpu(), b
        diff = (a_cpu.float() - b_cpu.float()).abs()
        tol = 2e-2 + 2e-2 * b_cpu.float().abs()
        bad = diff > tol
        bad_frac = bad.float().mean().item()
        # Fused low-precision kernels can disagree with the FP32 CPU ref on a
        # vanishingly small fraction of elements that sit exactly on a BF16
        # rounding boundary (different FP32 reduction order on GPU vs CPU).
        # Allow a tiny fraction of such boundary flips instead of requiring
        # bitwise-tight agreement everywhere.
        assert bad_frac == 0, (
            f"{name} mismatch: {bad.sum().item()}/{bad.numel()} elements "
            f"({bad_frac:.6f}) exceed atol=2e-2, rtol=2e-2 "
            f"(max abs diff={diff.max().item():.6f})"
        )


def mhc_pre_broadcast_ref(
    residual: torch.Tensor,        # (T, H), bfloat16
    fn_broadcast: torch.Tensor,    # (hc_mult3, H), float32
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor,
    norm_eps: float = 1e-6,
    hc_mult: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference, mirroring the fused Triton kernel's exact rounding
    order (sqrsum of the BF16-rounded collapsed value, not the FP32 value).
    """

    num_tokens, hidden_size = residual.shape
    residual_f32 = residual.float()
    sqrsum = hc_mult * residual_f32.square().sum(-1)  # 4 identical copies
    proj = (residual_f32 @ fn_broadcast.to(torch.bfloat16).float().T).bfloat16().float()
    mixes = proj * (
        sqrsum.unsqueeze(-1) / (hc_mult * hidden_size) + rms_eps
    ).rsqrt()

    hc_scale_expanded = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * hc_scale_expanded + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid() + hc_pre_eps  # (T, hc_mult)
    pre_sum = pre_mix.sum(-1, keepdim=True)              # (T, 1)

    post_mix = (mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult)
    res_mix = sinkhorn_normalize_ref(res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps)

    # Collapse, round to BF16 (matches kernel's output_shared boundary),
    # THEN compute sqrsum from the rounded value.
    raw = pre_sum * residual_f32
    raw_bf16 = raw.bfloat16()
    raw_rounded = raw_bf16.float()
    sumsq = raw_rounded.square().sum(-1, keepdim=True)
    norm_inv = (sumsq / hidden_size + norm_eps).rsqrt()
    layer_input = (raw_rounded * norm_inv * norm_weight.float()).bfloat16()

    residual_out = residual.unsqueeze(1).expand(num_tokens, hc_mult, hidden_size).contiguous()

    return residual_out, post_mix, res_mix, layer_input

