# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from flaggems_vllm import gdn_w8a16


@pytest.fixture(autouse=True)
def ieee_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def reference(q, k, v, g, beta, state, bounds):
    h = state.transpose(-1, -2).clone()
    out = torch.empty_like(v, dtype=torch.float32)
    group = v.shape[2] // q.shape[2]
    q = q[0].float().repeat_interleave(group, 1).transpose(0, 1)
    k = k[0].float().repeat_interleave(group, 1).transpose(0, 1)
    v = v[0].float().transpose(0, 1)
    g = g[0].transpose(0, 1)
    beta = beta[0].transpose(0, 1)
    scale = q.shape[-1] ** -0.5
    for seq, (start, end) in enumerate(zip(bounds, bounds[1:])):
        s = h[seq]
        for pos in range(start, end, 64):
            stop = min(pos + 64, end)
            qq, kk, vv = q[:, pos:stop], k[:, pos:stop], v[:, pos:stop]
            gg = g[:, pos:stop].cumsum(-1)
            bb = beta[:, pos:stop]
            decay = (gg[:, :, None] - gg[:, None, :]).exp().tril()
            lhs = (kk @ kk.transpose(-1, -2) * decay * bb[:, :, None]).tril(-1)
            lhs.diagonal(dim1=-2, dim2=-1).fill_(1)
            rhs = bb[:, :, None] * (vv - gg.exp()[:, :, None] * (kk @ s))
            u = torch.linalg.solve_triangular(lhs, rhs, upper=False, unitriangular=True)
            result = (
                gg.exp()[:, :, None] * (qq @ s)
                + (qq @ kk.transpose(-1, -2) * decay) @ u
            ) * scale
            out[0, pos:stop] = result.transpose(0, 1)
            s = gg[:, -1].exp()[:, None, None] * s + kk.transpose(-1, -2) @ (
                (gg[:, -1:] - gg).exp()[:, :, None] * u
            )
        h[seq] = s
    return out, h.transpose(-1, -2)


def inputs(row, dtype=torch.bfloat16):
    shape = row["shape"]
    packed = len(shape) == 4
    if packed:
        tokens, _, hv, dv = shape
        b, heads, dk, n = 1, 16, 128, tokens
    else:
        b, tokens, heads, dk, hv, dv, n = shape
    bounds = [i * tokens // n for i in range(n + 1)]
    cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    if packed:
        mixed = torch.randn(
            tokens, 2 * heads * dk + hv * dv, device="cuda", dtype=dtype
        )
        q = mixed[:, : heads * dk].view(1, tokens, heads, dk)
        k = mixed[:, heads * dk : 2 * heads * dk].view(1, tokens, heads, dk)
        v = mixed[:, 2 * heads * dk :].view(1, tokens, hv, dv)
        q.copy_(torch.nn.functional.normalize(q.float(), dim=-1).to(dtype))
        k.copy_(torch.nn.functional.normalize(k.float(), dim=-1).to(dtype))
    else:
        q = torch.nn.functional.normalize(
            torch.randn(b, tokens, heads, dk, device="cuda"), dim=-1
        ).to(dtype)
        k = torch.nn.functional.normalize(
            torch.randn(b, tokens, heads, dk, device="cuda"), dim=-1
        ).to(dtype)
        v = torch.randn(b, tokens, hv, dv, device="cuda", dtype=dtype)
    g = -torch.rand(b, tokens, hv, device="cuda") * 0.1
    beta = torch.rand(b, tokens, hv, device="cuda")
    state = torch.randn(n, hv, dv, dk, device="cuda") * 0.1
    return (q, k, v, g, beta, state, cu), bounds


def check(row, dtype):
    args, bounds = inputs(row, dtype)
    before = args[5].clone()
    out, state = gdn_w8a16(*args)
    ref_out, ref_state = reference(*args[:6], bounds)
    torch.testing.assert_close(out.float(), ref_out, atol=0.002, rtol=0.02)
    torch.testing.assert_close(state, ref_state, atol=1e-4, rtol=0.002)
    torch.testing.assert_close(args[5], before, atol=0, rtol=0)
    return (
        args,
        float((out.float() - ref_out).abs().max()) if out.numel() else 0.0,
        float((state - ref_state).abs().max()),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,sequences", [(0, 1), (1, 1), (17, 3), (129, 2)])
@pytest.mark.gdn_w8a16
def test_gdn_w8a16(dtype, tokens, sequences):
    torch.manual_seed(1234)
    check(
        dict(
            op="chunk_gated_delta_rule", shape=[1, tokens, 16, 128, 32, 128, sequences]
        ),
        dtype,
    )


@pytest.mark.gdn_w8a16
def test_gdn_w8a16_packed():
    torch.manual_seed(1234)
    check(dict(op="packed", shape=[7, 8192, 32, 128]), torch.bfloat16)


@pytest.mark.gdn_w8a16
def test_gdn_w8a16_reject_dtype():
    args, _ = inputs(dict(op="packed", shape=[1, 8192, 32, 128]))
    with pytest.raises(NotImplementedError):
        gdn_w8a16(args[0].float(), *args[1:])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,sequences", [(512, 1), (1025, 3)])
@pytest.mark.gdn_w8a16
def test_gdn_w8a16_chunk(dtype, tokens, sequences):
    torch.manual_seed(1234)
    check(dict(shape=[1, tokens, 16, 128, 32, 128, sequences]), dtype)


@pytest.mark.gdn_w8a16
def test_gdn_w8a16_chunk_graph_boundaries():
    torch.manual_seed(1234)
    args, bounds = inputs(dict(shape=[1, 513, 16, 128, 32, 128, 3]))
    before = args[5].clone()
    for _ in range(3):
        gdn_w8a16(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gdn_w8a16(*args)
    # Update contents at the same address: no identity-cached chunk metadata.
    bounds = [0, 0, 1, 513]
    args[-1].copy_(torch.tensor(bounds, device=args[0].device, dtype=torch.int32))
    graph.replay()
    expected = reference(*args[:6], bounds)
    torch.testing.assert_close(actual[0].float(), expected[0], atol=0.002, rtol=0.02)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=0.002)
    torch.testing.assert_close(args[5], before, atol=0, rtol=0)


@pytest.mark.gdn_w8a16
def test_gdn_w8a16_single_dispatch_nonuniform():
    # T == N does not imply one token per sequence.
    args, _ = inputs(dict(shape=[1, 4, 16, 128, 32, 128, 4]))
    bounds = [0, 0, 1, 1, 4]
    args[-1].copy_(torch.tensor(bounds, device=args[0].device, dtype=torch.int32))
    actual = gdn_w8a16(*args)
    expected = reference(*args[:6], bounds)
    torch.testing.assert_close(actual[0].float(), expected[0], atol=0.002, rtol=0.02)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=0.002)


@pytest.mark.gdn_w8a16
def test_gdn_w8a16_chunk_packed_custom_scale():
    torch.manual_seed(1234)
    args, bounds = inputs(dict(shape=[1, 513, 16, 128, 32, 128, 3]))
    q, k, v, g, beta, state, cu = args
    mixed = torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), dim=-1)
    q = mixed[..., :2048].view_as(q)
    k = mixed[..., 2048:4096].view_as(k)
    v = mixed[..., 4096:].view_as(v)
    actual = gdn_w8a16(q, k, v, g, beta, state, cu, scale=0.25)
    expected = reference(*args[:6], bounds)
    torch.testing.assert_close(
        actual[0].float(), expected[0] * (0.25 / 128**-0.5), atol=0.002, rtol=0.02
    )
    torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=0.002)


@pytest.mark.parametrize("sequences", [3, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.gdn_w8a16
def test_gdn_w8a16_bounded_workspace_replay(monkeypatch, dtype, sequences):
    from flaggems_vllm import runtime

    if runtime.device.vendor_name != "metax":
        pytest.skip("MetaX windowed chunk implementation")
    import flaggems_vllm.runtime.backend._metax.gdn_w8a16_chunk as chunk

    # A small budget forces several windows without a large CI allocation.
    budget = 4 * 1024**2
    monkeypatch.setattr(chunk, "_WORKSPACE_LIMIT_BYTES", budget)
    args, _ = inputs(dict(shape=[1, 641, 2, 128, 4, 128, sequences]), dtype)
    original = args[5].clone()
    for _ in range(3):
        gdn_w8a16(*args)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = gdn_w8a16(*args)
    torch.cuda.synchronize()
    result_bytes = sum(x.numel() * x.element_size() for x in actual)
    assert torch.cuda.max_memory_allocated() - allocated <= budget + result_bytes + 4096
    del actual
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gdn_w8a16(*args)
    # Move empty and long requests across windows at the same Cu address.
    for bounds in (
        [0] * (sequences - 1) + [1, 641],
        [0, 321] + [321] * (sequences - 2) + [641],
    ):
        args[-1].copy_(torch.tensor(bounds, device="cuda", dtype=torch.int32))
        expected = reference(*args[:6], bounds)
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(
                actual[0].float(), expected[0], atol=0.002, rtol=0.02
            )
            torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=0.002)
            torch.testing.assert_close(args[5], original, atol=0, rtol=0)
