# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Accuracy tests for mHC (Manifold Constrained Hyper-Connection) operators.

Tests both mhc_post and mhc_pre against PyTorch reference implementations,
and optionally compares with TileLang implementations.
"""

from itertools import product

import pytest
import torch

import flaggems_vllm

try:
    from vllm.model_executor.layers.mhc import (
        _hc_head_fused_kernel as _vllm_hc_head_fused,
    )

    HAS_VLLM = True
except (ImportError, AttributeError):
    HAS_VLLM = False

# Every operator is taken from the package namespace, never from its module path:
# a path import bypasses the vendor dispatch this suite is meant to exercise (on
# Ascend it runs the generic Triton kernel instead of the specialised one).
hc_head_fused_kernel = flaggems_vllm.hc_head_fused_kernel
hc_split_sinkhorn = flaggems_vllm.hc_split_sinkhorn
mhc_bwd = flaggems_vllm.mhc_bwd
MixLayout = flaggems_vllm.MixLayout
mhc_post = flaggems_vllm.mhc_post
mhc_pre = flaggems_vllm.mhc_pre


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation of mhc_pre."""
    hc_mult = residual.shape[-2]
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = (
        residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    )
    hc_scale_expanded = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * hc_scale_expanded + hc_base
    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
    res_mix = sinkhorn_normalize_ref(
        res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps
    )
    layer_input = (residual * pre_mix).sum(-2).bfloat16()
    return post_mix, res_mix, layer_input


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------
# These are the PyTorch references the operator modules used to carry, both
# for tests and as silent runtime fallbacks. They now live here: production
# code must not fall back to torch compute (NO_TORCH_COMPUTE_FALLBACK), so
# the unsupported paths raise NotImplementedError instead. `benchmark/`
# imports them from this module.
EPS = 1e-10  # step-size guard inside the CG reference below


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    *,
    mix_layout: MixLayout = MixLayout.TRANSPOSED,
) -> torch.Tensor:
    """PyTorch reference implementation (mirrors :func:`mhc_post` layouts)."""
    if not isinstance(mix_layout, MixLayout):
        raise TypeError(
            "mix_layout must be a MixLayout enum member, "
            f"got {type(mix_layout).__name__}"
        )
    mix = comb_res_mix.mT if mix_layout is MixLayout.TRANSPOSED else comb_res_mix
    y = x.unsqueeze(-2) * post_layer_mix + torch.bmm(mix, residual.float())
    return y.type_as(x)


def mhc_bwd_ref(
    out: torch.Tensor,
    dout: torch.Tensor,
    cg_iters: int = None,
) -> torch.Tensor:
    """PyTorch reference implementation of Sinkhorn backward via implicit CG.

    Args:
        out: Sinkhorn output R, shape (seqlen, n_stream, n_stream), float32.
        dout: Upstream gradient dR, same shape as out, float32.
        cg_iters: Number of CG iterations. Defaults to 2 * n_stream.

    Returns:
        Gradient w.r.t. pre-Sinkhorn input, same shape as out.
    """
    seqlen, n_stream, _ = out.shape
    if cg_iters is None:
        cg_iters = 2 * n_stream

    R = out.float()
    dR = dout.float()

    # RdR = R * dR
    RdR = R * dR

    # b1 = sum(RdR, dim=-1), b2 = sum(RdR, dim=-2)
    b1 = RdR.sum(dim=-1)  # (seqlen, n_stream)
    b2 = RdR.sum(dim=-2)  # (seqlen, n_stream)

    # Initialize CG
    x1 = torch.zeros_like(b1)
    x2 = torch.zeros_like(b2)

    # Diagonal blocks: row and column sums of R (the identity only when R is
    # doubly stochastic).
    row_sum = R.sum(dim=-1)
    col_sum = R.sum(dim=-2)

    def matvec(r, x1_in, x2_in):
        # y1[i] = sum_j(R[i,j] * x2[j]) + rowsum(R)[i] * x1[i]
        y1 = (r * x2_in.unsqueeze(-2)).sum(dim=-1) + row_sum * x1_in
        # y2[j] = sum_i(R[i,j] * x1[i]) + colsum(R)[j] * x2[j]
        y2 = (r * x1_in.unsqueeze(-1)).sum(dim=-2) + col_sum * x2_in
        return y1, y2

    # r = b - A*x (with x=0, r = b)
    r1, r2 = b1.clone(), b2.clone()
    p1, p2 = r1.clone(), r2.clone()
    r_normsq = (r1 * r1 + r2 * r2).sum(dim=-1)  # (seqlen,)

    for _ in range(cg_iters):
        # Ap = A * p
        Ap1, Ap2 = matvec(R, p1, p2)

        # pAp = dot(p, Ap)
        pAp = (p1 * Ap1 + p2 * Ap2).sum(dim=-1)  # (seqlen,)

        # alpha = r_normsq / (pAp + eps)
        alpha = r_normsq / (pAp + EPS)
        alpha = alpha.unsqueeze(-1)  # (seqlen, 1)

        # x = x + alpha * p
        x1 = x1 + alpha * p1
        x2 = x2 + alpha * p2

        # r = r - alpha * Ap
        r1 = r1 - alpha * Ap1
        r2 = r2 - alpha * Ap2

        # r_new_normsq = dot(r, r)
        r_new_normsq = (r1 * r1 + r2 * r2).sum(dim=-1)

        # beta = r_new_normsq / (r_normsq + eps)
        beta = r_new_normsq / (r_normsq + EPS)
        beta = beta.unsqueeze(-1)

        # p = r + beta * p
        p1 = r1 + beta * p1
        p2 = r2 + beta * p2

        r_normsq = r_new_normsq

    # res = (dR - x1 - x2) * R
    res = (dR - x1.unsqueeze(-1) - x2.unsqueeze(-2)) * R
    return res


def sinkhorn_forward_ref(
    M: torch.Tensor, iters: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sinkhorn normalization forward pass.

    Args:
        M: Input logits, shape (..., n, n).
        iters: Number of Sinkhorn iterations.

    Returns:
        (R, P) where P = exp(M) and R is the doubly-stochastic matrix.
    """
    P = torch.exp(M)
    R = P.clone()
    for _ in range(iters):
        R = R / R.sum(-2, keepdim=True)
        R = R / R.sum(-1, keepdim=True)
    return R, P


def hc_head_fused_kernel_ref(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> torch.Tensor:
    """Pure PyTorch reference implementation for correctness testing."""
    if hs_flat.shape[0] == 0:
        return out
    x = hs_flat.reshape(hs_flat.shape[0], hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps
    result = torch.sum(pre_mix.unsqueeze(-1) * hs_flat.to(torch.float32), dim=1).to(
        out.dtype
    )
    out.copy_(result)
    return out


def mhc_split_sinkhorn_torch_ref(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outer_shape = mixes.shape[:-1]
    mix_hc = (2 + hc_mult) * hc_mult
    assert mixes.shape[-1] == mix_hc

    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].view(*outer_shape, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)

    row_max = comb.max(dim=-1, keepdim=True).values
    comb = (comb - row_max).exp()
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def generate_mhc_post_data(
    n: int, h: int, hc_mult: int = 4, device: str = flaggems_vllm.device
):
    torch.manual_seed(42)
    x = torch.randn((n, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n, hc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(
        (n, hc_mult, hc_mult), dtype=torch.float32, device=device
    )
    return dict(
        x=x,
        residual=residual,
        post_layer_mix=post_layer_mix,
        comb_res_mix=comb_res_mix,
    )


MHC_POST_CONFIGS = list(
    product(
        [4096],  # n (num_tokens)
        [1280, 2560, 7168],  # h (hidden_size)
        [2, 4],  # hc_mult
    )
)


@pytest.mark.mhc_post
@pytest.mark.parametrize(
    "n, h, hc_mult",
    MHC_POST_CONFIGS,
    ids=[f"n{n}_h{h}_hc{hc}" for n, h, hc in MHC_POST_CONFIGS],
)
def test_mhc_post_vs_ref(n, h, hc_mult):
    """Test Triton mhc_post against PyTorch CPU reference."""
    data = generate_mhc_post_data(n, h, hc_mult=hc_mult)
    out_triton = mhc_post(**data)
    data_cpu = {k: v.cpu() for k, v in data.items()}
    out_ref = mhc_post_ref(**data_cpu)
    torch.testing.assert_close(out_triton.cpu(), out_ref, rtol=1e-2, atol=1e-2)


MHC_POST_LAYOUT_CONFIGS = [
    (1, 3584, 4),
    (2, 3584, 4),
    (8, 3584, 4),
    (64, 3584, 4),
    (256, 3584, 4),
] + list(product([4096], [1280, 2560, 7168], [2, 4]))

# npu-only: exercise the generic (HC != 4) kernel at small N. Grids like
# (8, 2, 28) = 448 programs stay far below the NPU 65,535-program launch
# cap, so these pass with or without any autotune pruning.
if flaggems_vllm.device == "npu":
    MHC_POST_LAYOUT_CONFIGS += [
        (8, 3584, 2),
        (256, 3584, 2),
        (8, 1280, 2),
    ]


def make_mhc_post_mix_in_form(
    comb_res_mix: torch.Tensor, mix_layout, mem: str
) -> torch.Tensor:
    """Return ``comb_res_mix`` in the requested (mix_layout, mem) representation.

    NATURAL is the model-native layout; TRANSPOSED is the legacy caller-side
    ``mix.transpose(-1, -2).contiguous()`` layout.  ``mem='contig'`` —
    contiguous storage of the logical values; ``mem='view'`` — a non-contiguous,
    zero-copy view holding the same logical values (NATURAL: sliced view of a
    wider tensor; TRANSPOSED: the transpose view itself instead of a
    ``.contiguous()`` copy).
    """
    if mem not in ("contig", "view"):
        raise ValueError(f"mem must be 'contig' or 'view', got {mem!r}")
    if mix_layout is MixLayout.TRANSPOSED:
        transposed = comb_res_mix.transpose(-1, -2)
        return transposed.contiguous() if mem == "contig" else transposed
    if mem == "contig":
        return comb_res_mix
    n, h, w = comb_res_mix.shape
    wide = torch.randn(
        n, h, w + 3, dtype=comb_res_mix.dtype, device=comb_res_mix.device
    )
    wide[:, :, :w] = comb_res_mix
    return wide[:, :, :w]


@pytest.mark.mhc_post
@pytest.mark.parametrize(
    "n, h, hc_mult",
    MHC_POST_LAYOUT_CONFIGS,
    ids=[f"n{n}_h{h}_hc{hc}" for n, h, hc in MHC_POST_LAYOUT_CONFIGS],
)
@pytest.mark.parametrize(
    "mix_layout",
    [MixLayout.NATURAL, MixLayout.TRANSPOSED],
    ids=[lay.value for lay in (MixLayout.NATURAL, MixLayout.TRANSPOSED)],
)
@pytest.mark.parametrize("mem", ["contig", "view"])
def test_mhc_post_layout_matrix_vs_ref(n, h, hc_mult, mix_layout, mem):
    """Regression matrix: 2 layouts x 2 memory forms all match mhc_post_ref."""
    data = generate_mhc_post_data(n, h, hc_mult=hc_mult)
    mix = make_mhc_post_mix_in_form(data["comb_res_mix"], mix_layout, mem)
    # zero-copy contract: the view form must really be non-contiguous
    assert mix.is_contiguous() == (mem == "contig")
    data["comb_res_mix"] = mix
    out_triton = mhc_post(**data, mix_layout=mix_layout)
    data_cpu = {k: v.cpu() for k, v in data.items()}
    out_ref = mhc_post_ref(**data_cpu, mix_layout=mix_layout)
    if h == 3584:
        torch.testing.assert_close(out_triton.cpu(), out_ref, rtol=1e-2, atol=0.03125)
    else:
        torch.testing.assert_close(out_triton.cpu(), out_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.mhc_post
@pytest.mark.parametrize(
    "n, h, hc_mult",
    MHC_POST_LAYOUT_CONFIGS,
    ids=[f"n{n}_h{h}_hc{hc}" for n, h, hc in MHC_POST_LAYOUT_CONFIGS],
)
@pytest.mark.parametrize(
    "mix_layout",
    [MixLayout.NATURAL, MixLayout.TRANSPOSED],
    ids=[lay.value for lay in (MixLayout.NATURAL, MixLayout.TRANSPOSED)],
)
def test_mhc_post_layout_representation_invariant(n, h, hc_mult, mix_layout):
    """Contiguous vs view representation of identical values -> bitwise-equal output."""
    data = generate_mhc_post_data(n, h, hc_mult=hc_mult)
    base = data["comb_res_mix"]
    out_contig = mhc_post(
        **{
            **data,
            "comb_res_mix": make_mhc_post_mix_in_form(base, mix_layout, "contig"),
        },
        mix_layout=mix_layout,
    )
    out_view = mhc_post(
        **{
            **data,
            "comb_res_mix": make_mhc_post_mix_in_form(base, mix_layout, "view"),
        },
        mix_layout=mix_layout,
    )
    assert torch.equal(out_contig, out_view), (
        f"representation invariance broken: layout={mix_layout.value} "
        f"n={n} h={h} hc={hc_mult}"
    )


def generate_mhc_split_sinkhorn_data(
    batch: int,
    seqlen: int,
    hc_mult: int = 4,
    device: str = flaggems_vllm.device,
):
    torch.manual_seed(42)
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn((batch, seqlen, mix_hc), dtype=torch.float32, device=device)
    hc_scale = torch.randn((3,), dtype=torch.float32, device=device) * 0.1
    hc_base = torch.randn((mix_hc,), dtype=torch.float32, device=device) * 0.1
    return dict(mixes=mixes, hc_scale=hc_scale, hc_base=hc_base, hc_mult=hc_mult)


MHC_SPLIT_SINKHORN_CONFIGS = [
    (8, 16, 4),
    (32, 64, 4),
    (128, 128, 4),
    (256, 256, 4),
    (8, 16, 2),
    (32, 64, 2),
    (128, 128, 2),
    (256, 256, 2),
]


@pytest.mark.hc_split_sinkhorn
@pytest.mark.hc_split_sinkhorn_forward
@pytest.mark.skipif(
    flaggems_vllm.vendor_name == "ascend",
    reason=(
        "hc_split_sinkhorn is not validated on Ascend: the hc_mult=4 kernel "
        "with num_tokens > 16384 (BLOCK_N=256, num_warps=8) fails to compile "
        "in BiShengIR (UB overflow), so the large-shape case cannot run there"
    ),
)
@pytest.mark.parametrize(
    "batch, seqlen, hc_mult",
    MHC_SPLIT_SINKHORN_CONFIGS,
    ids=[f"b{b}_s{s}_hc{hc}" for b, s, hc in MHC_SPLIT_SINKHORN_CONFIGS],
)
def test_mhc_split_sinkhorn(batch, seqlen, hc_mult):
    """Test FlagGems split+sinkhorn implementation against DV."""
    data = generate_mhc_split_sinkhorn_data(batch, seqlen, hc_mult)
    pre_triton, post_triton, comb_triton = hc_split_sinkhorn(**data)
    pre_dv, post_dv, comb_dv = mhc_split_sinkhorn_torch_ref(**data)

    torch.testing.assert_close(pre_triton, pre_dv, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(post_triton, post_dv, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(comb_triton, comb_dv, rtol=1e-4, atol=1e-4)


MHC_PRE_CONFIGS = list(
    product(
        [512, 1024, 2048, 8192],  # n
        [1280, 2560, 4096],  # hidden_size
        [2, 4],  # hc_mult
    )
)


def generate_mhc_pre_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    device: str = flaggems_vllm.device,
):
    torch.manual_seed(42)
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    residual = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
        .bfloat16()
    )
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

    return dict(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )


@pytest.mark.mhc_pre
@pytest.mark.parametrize(
    "n, hidden_size, hc_mult",
    MHC_PRE_CONFIGS,
    ids=[f"n{n}_h{h}_hc_mult{hc_mult}" for n, h, hc_mult in MHC_PRE_CONFIGS],
)
def test_mhc_pre_vs_ref(n, hidden_size, hc_mult):
    """Test Triton mhc_pre against PyTorch CPU reference."""
    data = generate_mhc_pre_data(n, hc_mult, hidden_size)
    post_triton, comb_triton, li_triton = mhc_pre(**data)
    data_cpu = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    post_ref, comb_ref, li_ref = mhc_pre_ref(**data_cpu)

    torch.testing.assert_close(post_triton.cpu(), post_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(comb_triton.cpu(), comb_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(li_triton.cpu(), li_ref, rtol=1e-2, atol=1e-2)


MHC_BWD_CONFIGS = list(
    product(
        [256, 1024, 4096, 65536],  # seqlen
        [4],  # n_stream (optimized kernel only supports n_stream=4)
        [20],  # sinkhorn_iters
    )
)


def generate_mhc_bwd_data(
    seqlen: int,
    n_stream: int,
    sinkhorn_iters: int = 20,
    device: str = flaggems_vllm.device,
):
    """Generate test data for mhc_bwd.

    Returns (R, dR) where R is Sinkhorn output and dR is upstream gradient.
    """
    torch.manual_seed(42)
    dist = torch.distributions.uniform.Uniform(0.0, 4.0)
    M = dist.sample((seqlen, n_stream, n_stream)).to(device)

    R, _P = sinkhorn_forward_ref(M, iters=sinkhorn_iters)
    dR = torch.randn_like(R)

    return dict(R=R.detach(), dR=dR, n_stream=n_stream)


@pytest.mark.mhc_bwd
@pytest.mark.parametrize(
    "seqlen, n_stream, sinkhorn_iters",
    MHC_BWD_CONFIGS,
    ids=[f"seq{s}_ns{ns}_it{it}" for s, ns, it in MHC_BWD_CONFIGS],
)
def test_mhc_bwd_vs_ref(seqlen, n_stream, sinkhorn_iters):
    """Test Triton mhc_bwd against PyTorch CPU reference."""
    data = generate_mhc_bwd_data(seqlen, n_stream, sinkhorn_iters)
    R, dR = data["R"], data["dR"]

    out_triton = mhc_bwd(R, dR)
    out_ref = mhc_bwd_ref(R.cpu(), dR.cpu())

    torch.testing.assert_close(out_triton.cpu(), out_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.mhc_bwd
def test_mhc_bwd_unconverged():
    """Test mhc_bwd against float64 autograd through the 20-step Sinkhorn forward.

    After 20 iterations only the rows of R sum to 1, so the CG system must use the
    row and column sums of R. The error left is the difference between the implicit
    gradient and the gradient of the 20-step map (max 0.29 here); with identity
    diagonal blocks the max is 43.
    """
    torch.manual_seed(0)
    M = torch.randn(16384, 4, 4, dtype=torch.float64, device=flaggems_vllm.device) * 2
    M.requires_grad_(True)
    R, _P = sinkhorn_forward_ref(M, iters=20)
    dR = torch.randn_like(R)
    (R * dR).sum().backward()

    out = mhc_bwd(R.detach().float(), dR.float()).double()
    err = (out - M.grad).flatten(1).norm(dim=-1) / M.grad.flatten(1).norm(dim=-1)
    assert err.max().item() < 1.0, err.max().item()


MHC_HC_HEAD_FUSED_CONFIGS = [
    (1, 1280, 4),
    (4, 2560, 4),
    (16, 4096, 4),
    (64, 7168, 4),
    (256, 1280, 2),
    (256, 1280, 4),
    (512, 1280, 2),
    (512, 1280, 4),
    (512, 2560, 2),
    (512, 2560, 4),
    (1024, 2560, 2),
    (1024, 2560, 4),
    (2048, 4096, 2),
    (2048, 4096, 4),
    (4096, 1280, 2),
    (4096, 1280, 4),
]


def generate_hc_head_fused_data(
    n: int,
    hidden_size: int,
    hc_mult: int,
    dtype: torch.dtype,
    device: str = flaggems_vllm.device,
):
    torch.manual_seed(42)
    hs_flat = torch.randn((n, hc_mult, hidden_size), dtype=dtype, device=device)
    fn = torch.randn(
        (hc_mult, hc_mult * hidden_size), dtype=torch.float32, device=device
    )
    hc_scale = torch.randn((1,), dtype=torch.float32, device=device) * 0.1
    hc_base = torch.randn((hc_mult,), dtype=torch.float32, device=device) * 0.1
    out = torch.empty((n, hidden_size), dtype=dtype, device=device)
    return dict(
        hs_flat=hs_flat,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        out=out,
        hidden_size=hidden_size,
        rms_eps=1e-6,
        hc_eps=1e-6,
        hc_mult=hc_mult,
    )


@pytest.mark.hc_head_fused_kernel
@pytest.mark.parametrize(
    "n, hidden_size, hc_mult",
    MHC_HC_HEAD_FUSED_CONFIGS,
    ids=[f"n{n}_h{h}_hc{hc}" for n, h, hc in MHC_HC_HEAD_FUSED_CONFIGS],
)
def test_hc_head_fused_kernel_vs_ref(n, hidden_size, hc_mult):
    dtype = torch.bfloat16
    data = generate_hc_head_fused_data(n, hidden_size, hc_mult, dtype=dtype)
    data_ref = generate_hc_head_fused_data(n, hidden_size, hc_mult, dtype=dtype)

    out_triton = hc_head_fused_kernel(**data)
    out_ref = hc_head_fused_kernel_ref(**data_ref)
    torch.testing.assert_close(out_triton, out_ref, rtol=2e-2, atol=2e-2)


def _hc_head_fused_kernel_ref(
    hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
):
    _vllm_hc_head_fused(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )
    return out


@pytest.mark.hc_head_fused_kernel
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not available")
@pytest.mark.parametrize(
    "n, hidden_size, hc_mult",
    MHC_HC_HEAD_FUSED_CONFIGS,
    ids=[f"n{n}_h{h}_hc{hc}" for n, h, hc in MHC_HC_HEAD_FUSED_CONFIGS],
)
def test_hc_head_fused_kernel_vs_vllm(n, hidden_size, hc_mult):
    dtype = torch.bfloat16
    data = generate_hc_head_fused_data(n, hidden_size, hc_mult, dtype=dtype)
    data_ref = generate_hc_head_fused_data(n, hidden_size, hc_mult, dtype=dtype)

    out_triton = hc_head_fused_kernel(**data)
    out_ref = _hc_head_fused_kernel_ref(**data_ref)
    torch.testing.assert_close(out_triton, out_ref, rtol=2e-2, atol=2e-2)
