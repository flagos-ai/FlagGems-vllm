# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from flaggems_vllm import gdn_w8a16, vendor_name
from tests.gdn_reference import recurrent_gdn_reference, torch_gdn_decode

pytestmark = pytest.mark.skipif(
    vendor_name != "metax", reason="MetaX FP32-state GDN contract"
)


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
    import importlib

    chunk = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.gdn_w8a16"
    )

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


@pytest.mark.gdn_w8a16
def test_gdn_w8a16_backend_registration():
    import flaggems_vllm
    from flaggems_vllm.runtime.backend._metax.ops.gdn_w8a16 import (
        gdn_w8a16 as metax_gdn,
    )

    assert flaggems_vllm.gdn_w8a16 is metax_gdn
    entries = [e for e in flaggems_vllm._FULL_CONFIG if e[0] == "gdn_w8a16"]
    assert entries == [("gdn_w8a16", metax_gdn)]


@pytest.mark.gdn_w8a16
@pytest.mark.parametrize("real_packed", [False, True])
def test_gdn_w8a16_graph_contract(real_packed):
    """Multi-step decode and mixed-precision prefill with dynamic graph metadata."""
    from types import SimpleNamespace

    low = SimpleNamespace(
        gdn_w8a16_chunk=lambda args, scale: gdn_w8a16(
            *args, scale=scale, mixed_precision=True
        )
    )

    def public_decode(qkv, a, b, al, bias, st, ids, out, normalize=True, **kwargs):
        q, k, v = qkv if isinstance(qkv, tuple) else (qkv, None, None)
        return gdn_w8a16(
            q,
            k,
            v,
            initial_state=st,
            decode=True,
            a=a,
            b=b,
            A_log=al,
            dt_bias=bias,
            state_indices=ids,
            out=out,
            normalize=normalize,
        )

    decode = SimpleNamespace(decode=public_decode)
    torch.manual_seed(537)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []

    def close(a, b, atol=0.002, rtol=0.02):
        torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)

    def ref_decode(q, k, v, a, b, al, bias, st, ids, norm):
        n, hv, dv = v.shape[1:]
        out = torch.empty((n, 1, hv, dv), dtype=v.dtype, device=v.device)
        torch_gdn_decode(
            (q, k, v),
            a,
            b,
            al,
            bias,
            q.shape[-1] ** -0.5,
            st,
            out,
            ids,
            norm,
            round_beta=True,
        )
        return out.view_as(v)

    cases = (
        [(n, 128, 128) for n in (1, 2, 3, 7, 9, 31)]
        if real_packed
        else [(0, 128, 128), (1, 128, 128), (9, 128, 128), (3, 33, 17)]
    )
    for dtype in (torch.bfloat16, torch.float16):
        for norm in (True,) if real_packed else (False, True):
            for n, dk, dv in cases:
                h, hv = (16, 32) if real_packed else (2, 4)
                q = torch.randn((1, n, h, dk), device="cuda", dtype=dtype) * 0.2
                k = torch.randn_like(q) * 0.2
                v = torch.randn((1, n, hv, dv), device="cuda", dtype=dtype) * 0.2
                a = torch.randn((n, hv), device="cuda", dtype=dtype)
                b = torch.randn_like(a)
                al = torch.randn(hv, device="cuda") * 0.1
                bias = torch.randn_like(al) * 0.1
                ids = torch.arange(n, 0, -1, device="cuda", dtype=torch.int64)
                if n > 1:
                    ids[-1] = -1 if real_packed else 0
                original = torch.randn((n + 2, hv, dv, dk), device="cuda") * 0.1
                mixed = torch.cat(
                    (q[0].flatten(1), k[0].flatten(1), v[0].flatten(1)), dim=1
                )
                for packed in (True,) if real_packed else (False, True):
                    for factor, fma in [(False, False)]:
                        st = original.clone()
                        ref = original.clone()
                        out = torch.empty_like(v)

                        def call():
                            return decode.decode(
                                mixed if packed else (q, k, v),
                                a,
                                b,
                                al,
                                bias,
                                st,
                                ids,
                                out,
                                normalize=norm,
                                factor=factor,
                                fma=fma,
                            )

                        call()
                        st.copy_(original)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            call()
                        st.copy_(original)
                        for step in range(32 if n else 1):
                            expected = ref_decode(
                                q, k, v, a, b, al, bias, ref, ids, norm
                            )
                            graph.replay()
                            close(out, expected.to(dtype))
                            close(st, ref, atol=1e-5, rtol=0.002)
                        # Replay must consume changed slot mappings, not cached metadata values.
                        if n > 1:
                            ids.copy_(torch.arange(1, n + 1, device="cuda"))
                            st.copy_(original)
                            ref.copy_(original)
                            expected = ref_decode(
                                q, k, v, a, b, al, bias, ref, ids, norm
                            )
                            graph.replay()
                            close(out, expected.to(dtype))
                            close(st, ref, atol=1e-5, rtol=0.002)
                            ids.copy_(torch.arange(n, 0, -1, device="cuda"))
                            ids[-1] = 0
                        records.append(
                            dict(
                                kind="decode",
                                dtype=str(dtype),
                                normalize=norm,
                                n=n,
                                dk=dk,
                                dv=dv,
                                packed=packed,
                                factor=factor,
                                fma=fma,
                                passed=True,
                            )
                        )
                        print("PASS", records[-1], flush=True)

    if real_packed:
        assert len(records) == 12
        return

    # Validate GPU mappings, window boundaries, empty and skewed requests, both dtypes.
    for dtype in (torch.bfloat16, torch.float16):
        for lengths in (
            [0],
            [17],
            [43],
            [109],
            [0, 65, 0, 2050],
            [1, 2049, 2],
            [64] * 33,
        ):
            n = len(lengths)
            t = sum(lengths)
            h, hv, dk, dv = 2, 4, 128, 128
            q = torch.randn((1, t, h, dk), device="cuda", dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn((1, t, hv, dv), device="cuda", dtype=dtype)
            g = -torch.rand((1, t, hv), device="cuda") * 0.1
            b = torch.rand_like(g) / 128
            st = torch.randn((n, hv, dv, dk), device="cuda") * 0.1
            bounds = [0]
            for length in lengths:
                bounds.append(bounds[-1] + length)
            cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
            original = st.clone()

            # Independent per-token FP32 recurrence, including empty requests.
            def reference():
                return recurrent_gdn_reference(
                    q, k, v, g, b, st, bounds, normalize=False, scale=dk**-0.5
                )

            expected = reference()
            if not t:
                actual = gdn_w8a16(q, k, v, g, b, st, cu)
            else:
                actual = low.gdn_w8a16_chunk((q, k, v, g, b, st, cu), dk**-0.5)
            close(actual[0], expected[0])
            close(actual[1], expected[1])
            torch.testing.assert_close(st, original, atol=0, rtol=0)
            if t:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = low.gdn_w8a16_chunk((q, k, v, g, b, st, cu), dk**-0.5)
                graph.replay()
                close(actual[0], expected[0])
                close(actual[1], expected[1])
                # Change request lengths while retaining metadata shape and total tokens.
                lengths = list(reversed(lengths))
                bounds = [0]
                for length in lengths:
                    bounds.append(bounds[-1] + length)
                cu.copy_(torch.tensor(bounds, device="cuda", dtype=cu.dtype))
                expected = reference()
                graph.replay()
                close(actual[0], expected[0])
                close(actual[1], expected[1])
            records.append(
                dict(kind="prefill", dtype=str(dtype), lengths=lengths, passed=True)
            )
            print("PASS", records[-1], flush=True)

    assert len(records) == 46


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.gdn_w8a16
def test_gdn_w8a16_small_packed_scale_and_dynamic_padding(n, dtype):
    torch.manual_seed(991 + n)
    x = torch.randn(n, 8192, device="cuda", dtype=dtype)
    a = torch.randn(n, 32, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    al = torch.randn(32, device="cuda") * 0.1
    bias = torch.randn_like(al) * 0.1
    base = torch.randn(n + 2, 32, 128, 128, device="cuda") * 0.1
    state = base.clone()
    ids = torch.arange(n, 0, -1, device="cuda", dtype=torch.int64)
    out = torch.empty(1, n, 32, 128, device="cuda", dtype=dtype)
    for scale in (0.37, -0.21, 0.0):

        def call():
            return gdn_w8a16(
                x,
                initial_state=state,
                decode=True,
                a=a,
                b=b,
                A_log=al,
                dt_bias=bias,
                state_indices=ids,
                out=out,
                scale=scale,
            )

        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = call()
        for padding in (False, True, False):
            ids.copy_(torch.arange(n, 0, -1, device="cuda", dtype=torch.int64))
            if padding:
                ids[-1] = -(2**40)
            state.copy_(base)
            expected = base.clone()
            ref = torch.empty_like(out)
            torch_gdn_decode(
                x,
                a,
                b,
                al,
                bias,
                scale,
                expected,
                ref.view(n, 1, 32, 128),
                ids,
                True,
                round_beta=True,
            )
            graph.replay()
            assert actual[0] is out and actual[1] is state
            torch.testing.assert_close(out, ref, atol=0.002, rtol=0.02)
            torch.testing.assert_close(state, expected, atol=1e-5, rtol=0.002)
