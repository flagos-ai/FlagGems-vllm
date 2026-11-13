# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent CPU arithmetic, packed outputs and read-only input contract."""

import pytest
import torch

import flaggems_vllm as gems
from baseline import hc_down_silu as workload

# Two BF16 epsilons allow reduction-order differences at rounding boundaries.
# Structured rounding tests below require bit-exact results instead.
RTOL = 2 * torch.finfo(torch.bfloat16).eps
ATOL = 2e-3


def reference(x, weight, rank, hc):
    a = x.detach().cpu().float()
    w = weight[: rank + hc].detach().cpu().float()
    b = (a @ w.T).to(torch.bfloat16)
    z = b[:, :rank].float() * (1.0 / hc)
    return (z * torch.sigmoid(z)).to(torch.bfloat16), b[:, rank : rank + hc]


def assert_result(actual, expected, x, weight, rank, hc, *, exact=False):
    assert isinstance(actual, tuple) and len(actual) == 2
    m, n = x.shape[0], weight.shape[0]
    for out, ref, width, offset in zip(actual, expected, (rank, hc), (0, rank)):
        assert out.shape == (m, width)
        assert out.dtype == torch.bfloat16 and out.device == x.device
        assert out.stride() == (n, 1)
        assert out.storage_offset() == offset
        host = out.detach().cpu()
        torch.testing.assert_close(torch.isnan(host), torch.isnan(ref))
        torch.testing.assert_close(torch.isposinf(host), torch.isposinf(ref))
        torch.testing.assert_close(torch.isneginf(host), torch.isneginf(ref))
        torch.testing.assert_close(
            host,
            ref,
            rtol=0 if exact else RTOL,
            atol=0 if exact else ATOL,
            equal_nan=True,
        )
    assert actual[0].untyped_storage()._cdata == actual[1].untyped_storage()._cdata
    for source in (x, weight):
        assert actual[0].untyped_storage()._cdata != source.untyped_storage()._cdata


def check(args, *, exact=False):
    x, weight, rank, hc = args
    before = (x.cpu().clone(), weight.cpu().clone())
    expected = reference(*args)
    actual = gems.hc_down_silu(*args)
    assert_result(actual, expected, *args, exact=exact)
    for inp, old in zip((x, weight), before):
        torch.testing.assert_close(inp.cpu(), old, rtol=0, atol=0, equal_nan=True)
    return actual


@pytest.mark.hc_down_silu
@pytest.mark.parametrize(
    "binding", workload.grid(), ids=lambda b: str(workload.key_shape(b))
)
def test_hc_down_silu(binding):
    args, _ = workload.build_inputs(binding, torch.bfloat16, gems.device)
    # Poison only unused weight rows: an accidental padded reduction must fail.
    args[1][args[2] + args[3] :].fill_(float("nan"))
    check(args)


@pytest.mark.hc_down_silu
@pytest.mark.parametrize("m", [1, 4, 5, 49])
@pytest.mark.parametrize(
    "variant",
    ["rounding", "cancel", "partial_rounding", "saturation", "zero", "nonfinite"],
)
def test_hc_down_silu_numerics(m, variant):
    k, rank, hc, n = 2051, 3, 1, 9
    x = torch.zeros((m, k), dtype=torch.bfloat16)
    w = torch.zeros((n, k), dtype=torch.bfloat16)
    if variant == "rounding":
        # 1 + 2^-8 is exactly halfway between BF16 1 and 1+2^-7.
        # Rounding after SiLU would produce a different BF16 activation.
        x[:, :2] = torch.tensor([1, 2**-8])
        w[: rank + hc, :2] = 1
    elif variant == "cancel":
        x[:, [0, 1024, 2048]] = torch.tensor([256, 1, -256], dtype=x.dtype)
        w[: rank + hc, [0, 1024, 2048]] = 1
    elif variant == "partial_rounding":
        # Interleaved K partitions can contain 256+1 and -256 separately.
        # Rounding either partial to BF16 before final reduction loses the 1.
        x[:, [0, 1536, 256]] = torch.tensor([256, 1, -256], dtype=x.dtype)
        w[: rank + hc, [0, 1536, 256]] = 1
    elif variant == "saturation":
        x[:, 0] = 1
        w[:4, 0] = torch.tensor([100, -100, 1, -80])
    elif variant == "nonfinite":
        x[:, 0] = 1
        w[:4, 0] = torch.tensor(
            [float("inf"), -float("inf"), float("nan"), float("inf")]
        )
    check((x.to(gems.device), w.to(gems.device), rank, hc), exact=True)


@pytest.mark.hc_down_silu
@pytest.mark.parametrize("m", [1, 5, 49])
def test_hc_down_silu_fp32_scaling(m):
    # An extra BF16 rounding after /hc changes the final BF16 SiLU here.
    x = torch.zeros((m, 2051), dtype=torch.bfloat16)
    w = torch.zeros((11, 2051), dtype=torch.bfloat16)
    x[:, 0] = 1
    w[:6, 0] = 0.125
    check((x.to(gems.device), w.to(gems.device), 3, 3), exact=True)


@pytest.mark.hc_down_silu
@pytest.mark.parametrize("row_stride", [0, 1, 47])
def test_hc_down_silu_offset_and_readonly_overlap(row_stride):
    k, rank, hc = 33, 7, 3
    gen = torch.Generator().manual_seed(77)
    x = torch.randn((200,), generator=gen).to(torch.bfloat16).to(gems.device)
    w = torch.randn((800,), generator=gen).to(torch.bfloat16).to(gems.device)
    check(
        (
            x.as_strided((4, k), (row_stride, 1), 7),
            w.as_strided((14, k), (row_stride, 1), 5),
            rank,
            hc,
        )
    )


@pytest.mark.hc_down_silu
@pytest.mark.parametrize(
    "kind",
    [
        "rank0",
        "ranknegative",
        "rankbool",
        "rankfloat",
        "hc0",
        "hcnegative",
        "hcbool",
        "hcfloat",
        "n",
        "k",
        "k0",
        "xdim",
        "wdim",
        "xdtype",
        "wdtype",
        "xstride",
        "wstride",
        "device",
    ],
)
def test_hc_down_silu_errors(kind):
    x = torch.empty((2, 16), dtype=torch.bfloat16, device=gems.device)
    w = torch.empty((8, 16), dtype=torch.bfloat16, device=gems.device)
    rank, hc = 4, 3
    if kind.startswith("rank"):
        rank = {"rank0": 0, "ranknegative": -1, "rankbool": True, "rankfloat": 4.0}[
            kind
        ]
    elif kind.startswith("hc"):
        hc = {"hc0": 0, "hcnegative": -1, "hcbool": True, "hcfloat": 3.0}[kind]
    elif kind == "n":
        w = w[:6]
    elif kind == "k":
        w = w[:, :15]
    elif kind == "k0":
        x, w = x[:, :0], w[:, :0]
    elif kind == "xdim":
        x = x.unsqueeze(0)
    elif kind == "wdim":
        w = w.unsqueeze(0)
    elif kind == "xdtype":
        x = x.float()
    elif kind == "wdtype":
        w = w.float()
    elif kind == "xstride":
        x, w = x[:, ::2], w[:, :8]
    elif kind == "wstride":
        x, w = x[:, :8], w[:, ::2]
    elif kind == "device":
        w = w.cpu()
    with pytest.raises((ValueError, TypeError)):
        gems.hc_down_silu(x, w, rank, hc)


@pytest.mark.hc_down_silu
def test_hc_down_silu_cpu_unsupported():
    with pytest.raises(NotImplementedError):
        gems.hc_down_silu(
            torch.empty((1, 8), dtype=torch.bfloat16),
            torch.empty((4, 8), dtype=torch.bfloat16),
            1,
            1,
        )
