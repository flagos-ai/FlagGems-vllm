"""Tests for chunk_dplr (DPLR fused kernel implementation)."""

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

import flaggems_vllm
from flaggems_vllm.ops.FLA import chunk_dplr

# Tolerance values for numerical comparison
# Higher tolerance needed for bf16/fp16 due to reduced precision in attention ops
TOLERANCE_OUTPUT = 0.007  # For output tensor comparison
TOLERANCE_STATE = 0.008  # For hidden state comparison


def _cuda_available() -> bool:
    return torch.cuda.is_available() and flaggems_vllm.device == "cuda"


pytestmark = [
    pytest.mark.chunk_dplr,
    pytest.mark.skipif(not _cuda_available(), reason="chunk_dplr tests require CUDA"),
]


def chunk_dplr_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    scale: float = None,
    chunk_size: int = 64,
    keep_dtype: bool = False,
):
    """Reference implementation for chunk DPLR delta rule."""
    if keep_dtype:
        compute_dtype = q.dtype
        q, k, v, a, b, gk = map(
            lambda x: x.transpose(1, 2).to(compute_dtype), (q, k, v, a, b, gk)
        )
    else:
        q, k, v, a, b, gk = map(
            lambda x: x.transpose(1, 2).to(torch.float), (q, k, v, a, b, gk)
        )
    BT = chunk_size
    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT

    if keep_dtype:
        q, k, v, a, b, gk = map(
            lambda x: F.pad(x, (0, 0, 0, pad_len)), [q, k, v, a, b, gk]
        )
    else:
        q, k, v, a, b, gk = map(
            lambda x: F.pad(x, (0, 0, 0, pad_len)).to(torch.float), [q, k, v, a, b, gk]
        )
    B, H, _, K, V = *q.shape, v.shape[-1]
    NT = q.shape[-2] // BT
    if scale is None:
        scale = K**-0.5
    q = q * scale

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S += initial_state

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, a, b, gk = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=BT), [q, k, v, a, b, gk]
    )
    gk_cumsum = gk.cumsum(-2)
    A_ab = torch.zeros(B, H, NT, BT, BT, dtype=q.dtype, device=q.device)
    A_qk = torch.zeros(B, H, NT, BT, BT, dtype=q.dtype, device=q.device)
    A_ak = torch.zeros(B, H, NT, BT, BT, dtype=q.dtype, device=q.device)
    A_qb = torch.zeros(B, H, NT, BT, BT, dtype=q.dtype, device=q.device)

    for i in range(BT):
        a_i = a[:, :, :, i, None]
        q_i = q[:, :, :, i, None]
        gk_i = gk_cumsum[:, :, :, i, None]
        mask = (torch.arange(BT) <= i).to(q.device)
        attn_i = (
            (gk_i - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float("-inf")).exp()
        )
        A_qk[:, :, :, i, :] = (q_i * k * attn_i).sum(-1).clone()
        A_qb[:, :, :, i, :] = (q_i * b * attn_i).sum(-1).clone()
        mask = (torch.arange(BT) < i).to(q.device)
        attn_i = (
            (gk_i - gk[:, :, :, i, None] - gk_cumsum)
            .masked_fill(~mask.unsqueeze(-1), float("-inf"))
            .exp()
        )
        A_ab[:, :, :, i, :] = (a_i * b * attn_i).sum(-1).clone()
        A_ak[:, :, :, i, :] = (a_i * k * attn_i).sum(-1).clone()

    A_ab = A_ab
    for i in range(1, BT):
        A_ab[..., i, :i] = A_ab[..., i, :i].clone() + (
            A_ab[..., i, :, None].clone() * A_ab[..., :, :i].clone()
        ).sum(-2)

    A_ab = A_ab + torch.eye(BT, dtype=q.dtype, device=q.device)
    u = A_ab @ (A_ak @ v)
    w = A_ab @ ((gk_cumsum - gk).exp() * a)

    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        q_i, k_i, v_i, u_i, w_i, b_i = (
            q[:, :, i],
            k[:, :, i],
            v[:, :, i],
            u[:, :, i],
            w[:, :, i],
            b[:, :, i],
        )
        v2_i = u_i + w_i @ S
        o_1 = A_qk[:, :, i] @ v_i
        o_2 = A_qb[:, :, i] @ v2_i
        o_3 = (q_i * gk_cumsum[:, :, i].exp()) @ S
        o[:, :, i] = o_1 + o_2 + o_3
        decay = (gk_cumsum[:, :, i, -1, None] - gk_cumsum[:, :, i]).exp()
        S = (
            S * gk_cumsum[:, :, i, -1, :, None].exp()
            + (k_i * decay).transpose(-1, -2) @ v_i
            + (b_i * decay).transpose(-1, -2) @ v2_i
        )

    S = None if output_final_state is False else S
    o = rearrange(o, "b h n c d -> b h (n c) d")
    o = o[:, :, :T].transpose(1, 2)
    return o, S


def get_abs_err(x, y):
    return (x - y).abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = x.detach().flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(name, ref, tri, ratio, err_atol=1e-6):
    """Assert tensors are close (same logic as FLA's assert_close)."""
    if ref is None and tri is None:
        return
    assert ref is not None and tri is not None

    abs_atol = get_abs_err(ref, tri)
    error_rate = get_err_ratio(ref, tri)
    msg = f"{name:>16} diff: {abs_atol:.6f} ratio: {error_rate:.6f}"

    if abs_atol <= err_atol:
        return

    assert not torch.isnan(ref).any(), f"{name}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{name}: NaN detected in tri"
    assert error_rate < ratio, msg


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "mask_p", "lowerbound", "dtype"),
    [
        pytest.param(
            *test, id="B{}-T{}-H{}-D{}-mask_p{}-lowerbound{}-dtype{}".format(*test)
        )
        for test in [
            # Basic cases
            (1, 128, 1, 64, 0, -5, torch.float16),
            (2, 1024, 4, 128, 0, -5, torch.float16),
            # Edge: very short sequence
            (1, 17, 2, 64, 0, -5, torch.float16),
            # Edge: very long sequence
            (1, 2048, 2, 64, 0, -5, torch.float16),
            # Sparse masking (simulates dropped activations)
            (2, 512, 4, 64, 0.5, -5, torch.float16),
            # Different lowerbound (less strict gate clipping)
            (2, 512, 4, 64, 0, -3, torch.float16),
            # bfloat16 dtype (for H100 inference)
            (2, 1024, 4, 128, 0, -5, torch.bfloat16),
            # Large batch (batch inference)
            (4, 512, 4, 64, 0, -5, torch.float16),
            # Many heads (MHA/GQA)
            (2, 512, 8, 64, 0, -5, torch.float16),
        ]
    ],
)
def test_fused_forward(B, T, H, D, mask_p, lowerbound, dtype):
    """Test fused forward-only implementation against reference."""
    torch.manual_seed(42)

    q = torch.randn(B, T, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, T, H, D, dtype=dtype, device="cuda")
    v = torch.randn(B, T, H, D, dtype=dtype, device="cuda")
    a = torch.rand(B, T, H, D, dtype=dtype, device="cuda")
    gk = torch.randn(B, T, H, D, dtype=torch.float, device="cuda")

    a = F.normalize(a, p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(gk)
    gk = gk * (torch.rand_like(gk) > mask_p)
    gk = gk.clamp(lowerbound, 0)

    h0 = torch.randn(B, H, D, D, dtype=torch.float, device="cuda")

    # Reference uses fp32 for numerical stability (even when testing bf16)
    ref, ref_ht = chunk_dplr_delta_rule_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=1.0,
        initial_state=h0.clone(),
        output_final_state=True,
        chunk_size=16,
        keep_dtype=False,
    )

    tri, tri_ht = chunk_dplr(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=1.0,
        initial_state=h0.clone(),
        output_final_state=True,
        chunk_size=16,
    )

    # Convert to float32 for comparison (handles bf16 case)
    assert_close("o", ref, tri.float(), TOLERANCE_OUTPUT)
    assert_close("ht", ref_ht, tri_ht.float(), TOLERANCE_STATE)


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "lowerbound", "cu_seqlens", "dtype"),
    [
        pytest.param(
            *test, id="H{}-D{}-mask_p{}-lowerbound{}-cu_seqlens{}-{}".format(*test)
        )
        for test in [
            # Basic cases
            (4, 64, 0, -5, [0, 256, 500, 1000], torch.float16),
            # Short sequence
            (4, 64, 0, -5, [0, 17], torch.float16),
            # Many short sequences
            (4, 64, 0, -5, [0, 20, 40, 60, 80, 100], torch.float16),
            # Uneven lengths
            (4, 64, 0, -5, [0, 50, 100, 500, 1500], torch.float16),
            # Long sequences
            (4, 128, 0, -5, [0, 1000, 2000], torch.float16),
            # With masking
            (4, 64, 0.5, -5, [0, 256, 500, 1000], torch.float16),
            # Different lowerbound
            (4, 64, 0, -3, [0, 256, 500, 1000], torch.float16),
            # bfloat16 dtype
            (4, 128, 0, -5, [0, 256, 500, 1000], torch.bfloat16),
            # More heads
            (8, 64, 0, -5, [0, 256, 500], torch.float16),
        ]
    ],
)
def test_fused_forward_varlen(H, D, mask_p, lowerbound, cu_seqlens, dtype):
    """Test fused forward with variable length sequences."""
    torch.manual_seed(42)

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]

    q = torch.randn(1, T, H, D, dtype=dtype, device="cuda")
    k = torch.randn(1, T, H, D, dtype=dtype, device="cuda")
    v = torch.randn(1, T, H, D, dtype=dtype, device="cuda")
    a = torch.rand(1, T, H, D, dtype=dtype, device="cuda")
    gk = torch.randn(1, T, H, D, dtype=torch.float, device="cuda")

    a = F.normalize(a, p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(gk)
    gk = gk * (torch.rand_like(gk) > mask_p)
    gk = gk.clamp(lowerbound, 0)

    h0 = torch.randn(N, H, D, D, dtype=torch.float, device="cuda")
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")

    tri, tri_ht = chunk_dplr(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=1.0,
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_tensor,
        chunk_size=16,
    )

    ref, ref_ht = [], []
    for i in range(N):
        r_i, rh_i = chunk_dplr_delta_rule_ref(
            q=q[:, cu_seqlens[i] : cu_seqlens[i + 1]].clone(),
            k=k[:, cu_seqlens[i] : cu_seqlens[i + 1]].clone(),
            v=v[:, cu_seqlens[i] : cu_seqlens[i + 1]].clone(),
            a=a[:, cu_seqlens[i] : cu_seqlens[i + 1]].clone(),
            b=b[:, cu_seqlens[i] : cu_seqlens[i + 1]].clone(),
            gk=gk[:, cu_seqlens[i] : cu_seqlens[i + 1]].clone(),
            scale=1.0,
            initial_state=h0[i : i + 1].clone(),
            output_final_state=True,
            chunk_size=16,
            keep_dtype=False,
        )
        ref.append(r_i)
        ref_ht.append(rh_i)

    ref = torch.cat(ref, dim=1)
    ref_ht = torch.cat(ref_ht, dim=0)

    # Convert to float32 for comparison (handles bf16 case)
    assert_close("o", ref, tri.float(), TOLERANCE_OUTPUT)
    assert_close("ht", ref_ht, tri_ht.float(), TOLERANCE_STATE)
