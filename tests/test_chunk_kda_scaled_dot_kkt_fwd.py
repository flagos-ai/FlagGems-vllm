# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm as gems

pytestmark = pytest.mark.chunk_kda_scaled_dot_kkt_fwd
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def reference(q, k, g, beta, scale, bt, lengths=None):
    q, k, g, beta = [x.detach().cpu().double() for x in (q, k, g, beta)]
    b, t, h, _ = q.shape
    a = torch.zeros((b, t, h, bt), dtype=torch.float64)
    aqk = torch.zeros_like(a)
    spans = (
        [(0, sum(lengths[:i]), sum(lengths[: i + 1])) for i in range(len(lengths))]
        if lengths is not None
        else [(ib, 0, t) for ib in range(b)]
    )
    for ib, start, end in spans:
        for begin in range(start, end, bt):
            size = min(bt, end - begin)
            ii = torch.arange(size)
            lower = ii[:, None] >= ii[None, :]
            for head in range(h):
                qs, ks, gs = [x[ib, begin : begin + size, head] for x in (q, k, g)]
                delta = gs[:, None, :] - gs[None, :, :]
                weights = torch.exp2(torch.where(lower[:, :, None], delta, 0.0))
                kk = (ks[:, None, :] * ks[None, :, :] * weights).sum(-1)
                qk = (qs[:, None, :] * ks[None, :, :] * weights).sum(-1)
                a[ib, begin : begin + size, head, :size] = torch.where(
                    ii[:, None] > ii[None, :],
                    beta[ib, begin : begin + size, head, None] * kk,
                    0.0,
                )
                aqk[ib, begin : begin + size, head, :size] = torch.where(
                    lower, qk * scale, 0.0
                )
    return a, aqk


def inputs(b, t, h, dk, dtype, bt, lengths=None, gate="normal"):
    gen = torch.Generator().manual_seed(2701)
    q = (torch.randn((b, t, h, dk), generator=gen) * 0.2).to(dtype)
    k = (torch.randn((b, t, h, dk), generator=gen) * 0.2).to(dtype)
    g = torch.empty((b, t, h, dk), dtype=torch.float32)
    beta = torch.rand((b, t, h), generator=gen)
    spans = (
        [(0, sum(lengths[:i]), sum(lengths[: i + 1])) for i in range(len(lengths))]
        if lengths is not None
        else [(ib, 0, t) for ib in range(b)]
    )
    for ib, start, end in spans:
        for begin in range(start, end, bt):
            n = min(bt, end - begin)
            increments = -torch.rand((n, h, dk), generator=gen) * 0.1
            cumulative = increments.cumsum(0)
            if gate == "zero":
                cumulative.zero_()
            elif gate == "offset":
                cumulative -= 2000.0
            elif gate == "strong":
                cumulative *= 1000.0
            g[ib, begin : begin + n] = cumulative
    return tuple(x.to(gems.device).contiguous() for x in (q, k, g, beta))


def check(args, bt, out_dtype, lengths=None, indexed=False, scale=0.37):
    kw = dict(chunk_size=bt, scale=scale, output_dtype=out_dtype)
    if lengths is not None:
        cu = [0]
        for n in lengths:
            cu.append(cu[-1] + n)
        kw["cu_seqlens"] = torch.tensor(cu, device=gems.device, dtype=torch.int64)
        if indexed:
            pairs = [
                (i, c) for i, n in enumerate(lengths) for c in range((n + bt - 1) // bt)
            ]
            kw["chunk_indices"] = torch.tensor(
                pairs, device=gems.device, dtype=torch.int32
            ).reshape(-1, 2)
    before = [x.clone() for x in args]
    actual = gems.chunk_kda_scaled_dot_kkt_fwd(*args, **kw)
    expected = reference(*args, scale, bt, lengths)
    tol = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[out_dtype]
    for got, want in zip(actual, expected):
        assert got.shape == want.shape and got.dtype == out_dtype
        torch.testing.assert_close(
            got.cpu().float(), want.to(out_dtype).float(), atol=tol, rtol=tol
        )
    for old, now in zip(before, args):
        torch.testing.assert_close(old, now, atol=0, rtol=0)
    assert actual[0].data_ptr() != actual[1].data_ptr()
    return actual, kw


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("out_dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,bt",
    [
        ((1, 1, 1, 1), 16),
        ((1, 15, 2, 15), 16),
        ((1, 17, 1, 31), 16),
        ((2, 33, 2, 64), 32),
        ((1, 63, 2, 65), 64),
        ((1, 64, 1, 128), 64),
        ((1, 65, 2, 255), 64),
        ((1, 129, 1, 256), 128),
    ],
)
def test_fixed(dtype, out_dtype, shape, bt):
    check(inputs(*shape, dtype, bt), bt, out_dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("lengths", [[0, 1, 17, 0, 65], [32, 32], [1, 1, 1]])
def test_packed(dtype, indexed, lengths):
    check(
        inputs(1, sum(lengths), 2, 65, dtype, 32, lengths),
        32,
        torch.float32,
        lengths,
        indexed,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gate", ["zero", "offset", "strong"])
def test_gate_extremes(dtype, gate):
    check(inputs(1, 65, 2, 64, dtype, 64, gate=gate), 64, torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("gate", ["normal", "offset", "strong"])
def test_long_path(dtype, packed, gate):
    lengths = [0, 17, 496] if packed else None
    args = inputs(1, 513, 2, 65, dtype, 64, lengths, gate)
    check(args, 64, torch.float32, lengths, indexed=packed)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gate_dtype", DTYPES)
@pytest.mark.parametrize("beta_dtype", DTYPES)
def test_auxiliary_dtypes(dtype, gate_dtype, beta_dtype):
    q, k, g, beta = inputs(1, 33, 2, 31, dtype, 32)
    check(
        (q, k, g.to(gate_dtype).contiguous(), beta.to(beta_dtype).contiguous()),
        32,
        torch.float32,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bt", [16, 32, 64, 128])
def test_long_batch_and_chunk_size(dtype, bt):
    check(inputs(2, 513, 1, 128, dtype, bt), bt, torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value", [0.0, 1.0, -1.0])
def test_beta_rows(dtype, value):
    q, k, g, beta = inputs(1, 17, 2, 16, dtype, 16)
    beta.fill_(value)
    check((q, k, g, beta), 16, torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
def test_nonfinite_query(dtype):
    args = inputs(1, 17, 1, 1, dtype, 16)
    args[0][0, 0, 0, 0] = float("nan")
    args[0][0, 1, 0, 0] = float("inf")
    actual = gems.chunk_kda_scaled_dot_kkt_fwd(*args, scale=0.37, chunk_size=16)
    expected = reference(*args, 0.37, 16)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(
            got.cpu().double(), want, equal_nan=True, atol=2e-5, rtol=2e-5
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gate", ["normal", "offset", "strong"])
def test_dense_packed_path(dtype, gate):
    lengths = [0, 17, 496]
    args = inputs(1, 513, 16, 65, dtype, 64, lengths, gate)
    check(args, 64, torch.float32, lengths, indexed=True)


def test_dense_batch_path():
    check(inputs(2, 513, 8, 128, torch.float32, 64), 64, torch.float32)


@pytest.mark.parametrize("name", ["scale", "chunk_size"])
def test_tensor_scalar_rejected(name):
    args = inputs(1, 1, 1, 16, torch.float16, 64)
    value = torch.tensor(64, device=gems.device)
    with pytest.raises(TypeError):
        gems.chunk_kda_scaled_dot_kkt_fwd(*args, **{name: value})


def test_defaults_and_scalars():
    args = inputs(1, 17, 2, 64, torch.float16, 64)
    got = gems.chunk_kda_scaled_dot_kkt_fwd(*args)
    expected = reference(*args, 64**-0.5, 64)
    for x, y in zip(got, expected):
        torch.testing.assert_close(x.cpu().double(), y, atol=2e-5, rtol=2e-5)
    check(args, 64, torch.float32, scale=0.0)


def test_partial_indices_and_empty():
    args = inputs(1, 65, 1, 16, torch.float32, 32)
    cu = torch.tensor([0, 65], device=gems.device, dtype=torch.int32)
    ci = torch.tensor([[0, 1]], device=gems.device, dtype=torch.int64)
    got = gems.chunk_kda_scaled_dot_kkt_fwd(
        *args, scale=0.37, cu_seqlens=cu, chunk_indices=ci, chunk_size=32
    )
    expected = reference(*args, 0.37, 32)
    for x, y in zip(got, expected):
        y[:, :32] = 0
        y[:, 64:] = 0
        torch.testing.assert_close(x.cpu().double(), y, atol=2e-5, rtol=2e-5)
    ci = torch.empty((0, 2), device=gems.device, dtype=torch.int32)
    got = gems.chunk_kda_scaled_dot_kkt_fwd(*args, cu_seqlens=cu, chunk_indices=ci)
    assert all(torch.count_nonzero(x).item() == 0 for x in got)
    empty = inputs(1, 0, 2, 64, torch.float16, 64)
    got = gems.chunk_kda_scaled_dot_kkt_fwd(*empty)
    assert all(x.shape == (1, 0, 2, 64) for x in got)


@pytest.mark.parametrize("bad", ["gate", "beta", "chunk", "shape", "stride", "grad"])
def test_unsupported(bad):
    args = list(inputs(1, 17, 2, 64, torch.float16, 64))
    kw = {}
    if bad == "gate":
        args[2] = None
    elif bad == "beta":
        args[3] = None
    elif bad == "chunk":
        kw["chunk_size"] = 48
    elif bad == "shape":
        args[1] = args[1][:, :1]
    elif bad == "stride":
        args[0] = args[0].transpose(-1, -2).contiguous().transpose(-1, -2)
    elif bad == "grad":
        args[0].requires_grad_()
    with pytest.raises((ValueError, NotImplementedError)):
        gems.chunk_kda_scaled_dot_kkt_fwd(*args, **kw)


def test_export():
    op = gems.chunk_kda_scaled_dot_kkt_fwd
    assert op is gems.ops.chunk_kda_scaled_dot_kkt_fwd
    assert ("chunk_kda_scaled_dot_kkt_fwd", op) in gems._FULL_CONFIG


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ci", [[[1, 1]], [[0, 0], [1, 2]], [[1, 0], [1, 7]]])
def test_dense_partial_indices(dtype, ci):
    lengths = [17, 496]
    args = inputs(1, 513, 16, 65, dtype, 64, lengths)
    cu = torch.tensor([0, 17, 513], device=gems.device, dtype=torch.int64)
    indices = torch.tensor(ci, device=gems.device, dtype=torch.int32)
    actual = gems.chunk_kda_scaled_dot_kkt_fwd(
        *args, cu_seqlens=cu, chunk_indices=indices, scale=0.37
    )
    expected = reference(*args, 0.37, 64, lengths)
    keep = torch.zeros(513, dtype=torch.bool)
    for seq, chunk in ci:
        start = sum(lengths[:seq]) + 64 * chunk
        keep[start : min(start + 64, sum(lengths[: seq + 1]))] = True
    for got, want in zip(actual, expected):
        want[:, ~keep] = 0
        torch.testing.assert_close(got.cpu().double(), want, atol=2e-5, rtol=2e-5)
