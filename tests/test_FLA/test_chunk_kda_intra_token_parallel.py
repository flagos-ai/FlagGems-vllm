# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
import itertools

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.FLA import chunk_kda_fwd_intra_token_parallel as op
from tests import conftest as cfg

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
TOLERANCES = {
    torch.float16: (2e-3, 2e-4),
    torch.bfloat16: (1e-2, 2e-3),
    torch.float32: (1e-4, 2e-6),
}
SENTINEL = -123.5
# (sequence lengths, packed varlen, H, HV, K, BT, BC)
CASES = [
    ((1,), False, 1, 1, 1, 64, 16),
    ((15, 15), False, 2, 2, 31, 64, 16),
    ((16,), False, 1, 3, 32, 64, 16),
    ((17,), False, 2, 6, 33, 64, 16),
    ((63,), False, 3, 3, 64, 64, 16),
    ((64,), False, 2, 4, 128, 64, 16),
    ((65,), False, 1, 5, 256, 64, 16),
    ((129,), False, 2, 4, 65, 64, 16),
    ((0, 1, 0, 17, 35, 0), True, 2, 6, 33, 64, 16),
    ((65, 3), True, 1, 3, 128, 64, 16),
    ((17,), False, 96, 96, 128, 64, 16),
    ((19, 19), False, 1, 2, 16, 32, 8),
    ((13, 25), True, 2, 4, 17, 12, 3),
    ((5,), False, 1, 1, 3, 1, 1),
]


def _buffer(shape, dtype):
    size = 1
    for dim in shape:
        size *= dim
    backing = torch.full(
        (size + 2,), SENTINEL, dtype=dtype, device=flaggems_vllm.device
    )
    return backing[1:-1].view(shape), backing


def _inputs(
    case, dtype, aux_dtype=torch.float32, output_dtype=None, index_dtype=torch.int64
):
    lengths, packed, H, HV, K, BT, BC = case
    B, T = (1, sum(lengths)) if packed else (len(lengths), lengths[0])
    generator = torch.Generator().manual_seed(20261026)
    q = torch.randn((B, T, H, K), generator=generator, dtype=torch.float64) * 0.1
    k = torch.randn(q.shape, generator=generator, dtype=torch.float64) * 0.1
    g = -torch.rand((B, T, HV, K), generator=generator, dtype=torch.float64) * 2
    beta = torch.rand((B, T, HV), generator=generator, dtype=torch.float64)
    q, k = (x.to(device=flaggems_vllm.device, dtype=dtype).contiguous() for x in (q, k))
    g, beta = (
        x.to(device=flaggems_vllm.device, dtype=aux_dtype).contiguous()
        for x in (g, beta)
    )
    a, a_back = _buffer((B, T, HV, BT), output_dtype or dtype)
    b, b_back = _buffer((B, T, HV, BC), torch.float32)
    cu = None
    if packed:
        cu = torch.tensor(
            [0, *itertools.accumulate(lengths)], dtype=index_dtype, device=q.device
        )
    return (q, k, g, beta, a, b), cu, (a_back, b_back)


def _reference(args, lengths, packed, BT, BC, scale):
    # Independent CPU FP64 pairwise formula, vectorized within each sub-block.
    q, k, g, beta, a, b = [x.detach().cpu().double() for x in args]
    H, HV = q.shape[2], g.shape[2]
    q = q.repeat_interleave(HV // H, dim=2)
    k = k.repeat_interleave(HV // H, dim=2)
    ma = torch.zeros_like(a, dtype=torch.bool)
    mb = torch.zeros_like(b, dtype=torch.bool)
    offset = 0
    for seq, length in enumerate(lengths):
        batch = 0 if packed else seq
        base = offset if packed else 0
        for start in range(0, length, BC):
            stop = min(start + BC, length)
            s = slice(base + start, base + stop)
            qi, ki, gi, bi = q[batch, s], k[batch, s], g[batch, s], beta[batch, s]
            weighted_keys = ki[None, :, :, :] * torch.exp2(gi[:, None] - gi[None, :])
            qk = (qi[:, None] * weighted_keys).sum(-1) * scale
            kk = ((ki * bi[..., None])[:, None] * weighted_keys).sum(-1)
            for i in range(stop - start):
                for j in range(i + 1):
                    a[batch, base + start + i, :, (start + j) % BT] = qk[i, j]
                    b[batch, base + start + i, :, j] = kk[i, j] * (
                        1.0 if j < i else 0.0
                    )
                    ma[batch, base + start + i, :, (start + j) % BT] = True
                    mb[batch, base + start + i, :, j] = True
        offset += length
    return a, b, ma, mb


def _check(
    case,
    dtype,
    aux_dtype=torch.float32,
    output_dtype=None,
    index_dtype=torch.int64,
    scale=None,
):
    args, cu, backing = _inputs(case, dtype, aux_dtype, output_dtype, index_dtype)
    lengths, packed, _, _, K, BT, BC = case
    scale = K**-0.5 if scale is None else scale
    before = [x.clone() for x in args[:4]]
    expected_a, expected_b, ma, mb = _reference(args, lengths, packed, BT, BC, scale)
    actual_a, actual_b = flaggems_vllm.chunk_kda_fwd_intra_token_parallel(
        *args, scale, cu_seqlens=cu, chunk_size=BT, sub_chunk_size=BC
    )
    assert actual_a is args[4] and actual_b is args[5]
    for actual, expected, mask in (
        (actual_a, expected_a, ma),
        (actual_b, expected_b, mb),
    ):
        actual = actual.cpu()
        rtol, atol = TOLERANCES[actual.dtype]
        torch.testing.assert_close(
            actual[mask],
            expected.to(actual.dtype)[mask],
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        torch.testing.assert_close(
            actual[~mask], expected.to(actual.dtype)[~mask], rtol=0, atol=0
        )
    for original, current in zip(before, args[:4]):
        torch.testing.assert_close(current, original, rtol=0, atol=0)
    for storage in backing:
        torch.testing.assert_close(
            storage[[0, -1]],
            torch.full_like(storage[[0, -1]], SENTINEL),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES[:3] if cfg.QUICK_MODE else CASES)
@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel(case, dtype):
    _check(case, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("aux_dtype", DTYPES)
@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel_aux_dtypes(dtype, aux_dtype):
    _check(
        ((0, 17, 0, 3), True, 2, 6, 33, 64, 16),
        dtype,
        aux_dtype,
        torch.float32,
        torch.int32,
        -0.25,
    )


@pytest.mark.parametrize(
    "case", [((0,), False, 1, 1, 32, 64, 16), ((0, 0), True, 1, 2, 32, 64, 16)]
)
@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel_empty(case):
    _check(case, torch.float32)


@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel_empty_batch():
    device = flaggems_vllm.device
    q = torch.empty((0, 17, 1, 32), device=device)
    g = torch.empty_like(q)
    beta = torch.empty((0, 17, 1), device=device)
    a = torch.empty((0, 17, 1, 64), device=device)
    b = torch.empty((0, 17, 1, 16), device=device)
    actual_a, actual_b = op(q, q, g, beta, a, b, 0.5)
    assert actual_a is a and actual_b is b


@pytest.mark.parametrize("a_dtype", DTYPES)
@pytest.mark.parametrize("b_dtype", DTYPES)
@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel_output_dtypes(a_dtype, b_dtype):
    case = ((17,), False, 1, 3, 65, 64, 16)
    args, _, _ = _inputs(case, torch.float32, output_dtype=a_dtype)
    args = (*args[:5], args[5].to(b_dtype))
    ea, eb, _, _ = _reference(args, case[0], False, 64, 16, 0.5)
    a, b = op(*args, 0.5)
    for actual, expected in ((a, ea), (b, eb)):
        rtol, atol = TOLERANCES[actual.dtype]
        torch.testing.assert_close(
            actual.cpu(), expected.to(actual.dtype), rtol=rtol, atol=atol
        )


@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel_no_grad():
    args, _, _ = _inputs(((1,), False, 1, 1, 32, 64, 16), torch.float32)
    args[0].requires_grad_()
    with torch.no_grad():
        actual_a, actual_b = op(*args, 0.5)
    assert actual_a is args[4] and actual_b is args[5]
    assert not actual_a.requires_grad and not actual_b.requires_grad


@pytest.mark.parametrize(
    "value", [0.0, -1000.0, float("nan"), float("inf"), -float("inf")]
)
@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel_gate_edges(value):
    case = ((3,), False, 1, 1, 3, 64, 16)
    args, cu, _ = _inputs(case, torch.float32)
    args[2].fill_(value)
    ea, eb, ma, mb = _reference(args, case[0], False, 64, 16, 0.5)
    a, b = op(*args, 0.5, cu_seqlens=cu)
    for actual, expected, mask in ((a, ea, ma), (b, eb, mb)):
        torch.testing.assert_close(
            actual.cpu(), expected.float(), rtol=1e-4, atol=2e-6, equal_nan=True
        )


@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel_zero_beta_scale_and_alias():
    case = ((17,), False, 1, 2, 32, 64, 16)
    args, _, _ = _inputs(case, torch.float32)
    q, _, g, beta, a, b = args
    beta.zero_()
    # q/k may share storage; output/input aliasing remains forbidden.
    args = (q, q, g, beta, a, b)
    ea, eb, ma, mb = _reference(args, case[0], False, 64, 16, 0.0)
    op(*args, 0.0)
    torch.testing.assert_close(a.cpu(), ea.float(), rtol=0, atol=0)
    torch.testing.assert_close(b.cpu(), eb.float(), rtol=0, atol=0)
    # Repeated execution overwrites the same domain without accumulating.
    op(*args, 0.0)
    torch.testing.assert_close(a.cpu(), ea.float(), rtol=0, atol=0)
    torch.testing.assert_close(b.cpu(), eb.float(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "invalid",
    [
        "q_rank",
        "k_shape",
        "g_shape",
        "beta_shape",
        "heads",
        "K",
        "qk_dtype",
        "integer",
        "noncontiguous",
        "output_shape",
        "output_dtype",
        "output_float64",
        "output_overlap",
        "input_overlap",
        "device",
        "cu_dtype",
        "cu_rank",
        "cu_short",
        "cu_batch",
        "cu_noncontiguous",
        "grad",
        "chunk_zero",
        "sub_zero",
        "chunk_divisibility",
        "chunk_type",
        "scale_nan",
        "graph",
        "cpu",
    ],
)
@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel_invalid(invalid):
    args, _, _ = _inputs(((17,), False, 2, 2, 32, 64, 16), torch.float32)
    args = list(args)
    kwargs = {"scale": 0.5}
    if invalid == "q_rank":
        args[0] = args[0][0]
    elif invalid == "k_shape":
        args[1] = args[1][:, :1]
    elif invalid == "g_shape":
        args[2] = args[2][:, :1]
    elif invalid == "beta_shape":
        args[3] = args[3][:, :1]
    elif invalid == "heads":
        args[2] = torch.empty((1, 17, 3, 32), device=args[0].device)
    elif invalid == "K":
        args[0] = args[1] = torch.empty((1, 17, 2, 257), device=args[0].device)
    elif invalid == "qk_dtype":
        args[1] = args[1].half()
    elif invalid == "integer":
        args[0] = args[0].int()
    elif invalid == "noncontiguous":
        args[0] = args[0].transpose(-1, -2)
    elif invalid == "output_shape":
        args[4] = args[4][:, :1]
    elif invalid == "output_dtype":
        args[5] = args[5].to(torch.int32)
    elif invalid == "output_float64":
        args[5] = args[5].double()
        if args[5].dtype != torch.float64:
            pytest.skip("device runtime coerces float64 tensors to float32")
    elif invalid == "output_overlap":
        args[5] = args[4].view(-1)[: args[5].numel()].view_as(args[5])
    elif invalid == "input_overlap":
        args[0] = args[4].view(-1)[: args[0].numel()].view_as(args[0])
    elif invalid == "device":
        args[5] = args[5].cpu()
    elif invalid.startswith("cu_"):
        cu = torch.tensor([0, 17], device=args[0].device)
        if invalid == "cu_dtype":
            cu = cu.float()
        elif invalid == "cu_rank":
            cu = cu[None]
        elif invalid == "cu_short":
            cu = cu[:1]
        elif invalid == "cu_noncontiguous":
            cu = torch.tensor([0, 0, 17, 17], device=cu.device)[::2]
        elif invalid == "cu_batch":
            args, _, _ = _inputs(((17, 17), False, 2, 2, 32, 64, 16), torch.float32)
        kwargs["cu_seqlens"] = cu
    elif invalid == "grad":
        args[0].requires_grad_()
    elif invalid == "chunk_zero":
        kwargs["chunk_size"] = 0
    elif invalid == "sub_zero":
        kwargs["sub_chunk_size"] = 0
    elif invalid == "chunk_divisibility":
        kwargs["chunk_size"] = 63
    elif invalid == "chunk_type":
        kwargs["chunk_size"] = 64.0
    elif invalid == "scale_nan":
        kwargs["scale"] = float("nan")
    elif invalid == "graph":
        kwargs["use_graph"] = True
    elif invalid == "cpu":
        args = [x.cpu() for x in args]
    with pytest.raises((ValueError, NotImplementedError)):
        op(*args, **kwargs)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.chunk_kda_fwd_intra_token_parallel
def test_chunk_kda_intra_token_parallel_fla_reference(dtype, packed):
    # Optional external cross-check; the independent oracle above is mandatory.
    pytest.importorskip("fla")
    native = importlib.import_module(
        "fla.ops.kda.chunk_intra_token_parallel"
    ).chunk_kda_fwd_intra_token_parallel
    case = ((1, 17, 35) if packed else (65,), packed, 2, 6, 33, 64, 16)
    args, cu, _ = _inputs(case, dtype)
    q, k, g, beta, a, b = args
    na, nb = a.clone(), b.clone()
    native(q, k, g, beta, na, nb, 0.5, cu_seqlens=cu, use_graph=False)
    op(*args, 0.5, cu_seqlens=cu)
    for actual, expected in ((a, na), (b, nb)):
        rtol, atol = TOLERANCES[actual.dtype]
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        )
