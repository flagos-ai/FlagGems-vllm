# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


repo = Path(os.getenv("GDN_REPO", Path(__file__).resolve().parents[1]))
backend_dir = repo / "src/flaggems_vllm/runtime/backend/_metax"
for module in ("gdn_w8a16_short", "gdn_w8a16_decode", "gdn_w8a16_fp32"):
    load(
        "flaggems_vllm.runtime.backend._metax." + module, backend_dir / (module + ".py")
    )
load("isolated_chunk", backend_dir / "gdn_w8a16_chunk.py")

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "entry.py"
    source = (
        (backend_dir / "ops/gdn_w8a16.py")
        .read_text()
        .replace(
            "from flaggems_vllm import runtime",
            'from types import SimpleNamespace\nruntime=SimpleNamespace(device=SimpleNamespace(vendor_name="metax"))',
        )
        .replace(
            "from flaggems_vllm.runtime.backend._metax.gdn_w8a16_chunk import",
            "from isolated_chunk import",
        )
    )
    path.write_text(source)
    base = load("isolated_entry", path)
low = SimpleNamespace(
    gdn_w8a16_chunk=lambda args, scale: base.gdn_w8a16(
        *args, scale=scale, mixed_precision=True
    )
)


def public_decode(qkv, a, b, al, bias, st, ids, out, normalize=True, **kwargs):
    q, k, v = qkv if isinstance(qkv, tuple) else (qkv, None, None)
    return base.gdn_w8a16(
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
    q = q[0].float()
    k = k[0].float()
    v = v[0].float()
    hv = st.shape[1]
    h = q.shape[1]
    if norm:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(hv // h, 1)
    k = k.repeat_interleave(hv // h, 1)
    valid = ids > 0
    slots = ids.clamp_min(0).long()
    s = st[slots]
    d = torch.exp(-al.exp() * torch.nn.functional.softplus(a.float() + bias))
    s = s * d[..., None, None]
    delta = (v - (s * k[..., None, :]).sum(-1)) * b.float().sigmoid().to(
        b.dtype
    ).float()[..., None]
    s = s + delta[..., None] * k[..., None, :]
    out = (s * q[..., None, :]).sum(-1) * q.shape[-1] ** -0.5
    out = torch.where(valid[:, None, None], out, 0)
    # Null slots must remain untouched; positive slots are distinct.
    st[ids[valid].long()] = s[valid]
    return out.unsqueeze(0)


for dtype in (torch.bfloat16, torch.float16):
    for norm in (False, True):
        for n, dk, dv in [(0, 128, 128), (1, 128, 128), (9, 128, 128), (3, 33, 17)]:
            h, hv = 2, 4
            q = torch.randn((1, n, h, dk), device="cuda", dtype=dtype) * 0.2
            k = torch.randn_like(q) * 0.2
            v = torch.randn((1, n, hv, dv), device="cuda", dtype=dtype) * 0.2
            a = torch.randn((n, hv), device="cuda", dtype=dtype)
            b = torch.randn_like(a)
            al = torch.randn(hv, device="cuda") * 0.1
            bias = torch.randn_like(al) * 0.1
            ids = torch.arange(n, 0, -1, device="cuda", dtype=torch.int64)
            if n > 1:
                ids[-1] = 0
            original = torch.randn((n + 2, hv, dv, dk), device="cuda") * 0.1
            mixed = torch.cat(
                (q[0].flatten(1), k[0].flatten(1), v[0].flatten(1)), dim=1
            )
            for packed in (False, True):
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
                        expected = ref_decode(q, k, v, a, b, al, bias, ref, ids, norm)
                        graph.replay()
                        close(out, expected.to(dtype))
                        close(st, ref, atol=1e-5, rtol=0.002)
                    # Replay must consume changed slot mappings, not cached metadata values.
                    if n > 1:
                        ids.copy_(torch.arange(1, n + 1, device="cuda"))
                        st.copy_(original)
                        ref.copy_(original)
                        expected = ref_decode(q, k, v, a, b, al, bias, ref, ids, norm)
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

# Validate GPU mappings, window boundaries, empty and skewed requests, both dtypes.
for dtype in (torch.bfloat16, torch.float16):
    for lengths in ([0], [17], [43], [109], [0, 65, 0, 2050], [1, 2049, 2], [64] * 33):
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
            out = torch.empty_like(v)
            final = []
            for seq in range(n):
                state = st[seq].clone()
                for token in range(bounds[seq], bounds[seq + 1]):
                    key = k[0, token].float().repeat_interleave(hv // h, 0)
                    query = q[0, token].float().repeat_interleave(hv // h, 0)
                    state = state * g[0, token].exp()[:, None, None]
                    delta = (
                        v[0, token].float() - (state * key[:, None, :]).sum(-1)
                    ) * b[0, token, :, None]
                    state += delta[:, :, None] * key[:, None, :]
                    out[0, token] = (state * query[:, None, :]).sum(-1) * dk**-0.5
                final.append(state)
            return out, torch.stack(final)

        expected = reference()
        if not t:
            actual = base.gdn_w8a16(q, k, v, g, b, st, cu)
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
Path(os.getenv("GDN_CHECK_RESULT", "gdn_checks.json")).write_text(
    json.dumps(records, indent=2)
)
print("DONE", len(records), flush=True)
