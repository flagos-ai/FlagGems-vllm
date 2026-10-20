# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import flaggems_vllm as fgv
import pytest
import torch
from flaggems_vllm.ops.fused_moe import MoEActivation, apply_moe_activation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def assert_fp32(actual, expected):
    a = actual.detach().cpu().float()
    b = expected.detach().cpu().float()
    d = a - b
    limit = 0.004 if actual.dtype in (torch.bfloat16, torch.float16) else 2e-6
    peak = 0.01 if actual.dtype in (torch.bfloat16, torch.float16) else 5e-6
    assert torch.isfinite(a).all()
    assert (d.norm() / b.norm().clamp_min(1e-20)).item() <= limit
    assert (d.abs().max() / b.abs().max().clamp_min(1e-20)).item() <= peak


@pytest.mark.parametrize(
    "m,n,dtype",
    [
        (1, 128, torch.bfloat16),
        (7, 6144, torch.bfloat16),
        (64, 6144, torch.bfloat16),
        (3, 257, torch.float16),
        (8, 128, torch.float32),
    ],
)
def test_gemma_residual_mode(m, n, dtype):
    torch.manual_seed(560)
    xb = torch.randn(m, n + 8, device="cuda", dtype=dtype)
    rb = torch.randn_like(xb)
    x, r = xb[:, :n], rb[:, :n]
    w = torch.randn(n, device="cuda", dtype=dtype)
    added = x.cpu().float() + r.cpu().float()
    tailx, tailr = xb[:, n:].clone(), rb[:, n:].clone()
    ref = (
        added
        * torch.rsqrt((added * added).sum(-1, keepdim=True) / n + 1e-6)
        * (1 + w.cpu().float())
    )
    out, res = fgv.fused_add_rms_norm(x, r, (n,), w, 1e-6, gemma=True)
    assert out is x and res is r
    assert_fp32(x, ref)
    assert torch.equal(r.cpu(), added.to(dtype))
    assert torch.equal(tailx, xb[:, n:]) and torch.equal(tailr, rb[:, n:])


def test_gemma_empty_and_alias():
    x = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    r = torch.empty_like(x)
    w = torch.ones(128, device="cuda", dtype=x.dtype)
    assert fgv.fused_add_rms_norm(x, r, (128,), w, gemma=True)[0] is x
    x = torch.ones(2, 128, device="cuda", dtype=w.dtype)
    with pytest.raises(ValueError):
        fgv.fused_add_rms_norm(x, x, (128,), w, gemma=True)


@pytest.mark.parametrize(
    "m,h", [(0, 192), (1, 127), (7, 192), (32, 768), (1024, 768), (8192, 192)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_division_swiglu(m, h, dtype):
    torch.manual_seed(560)
    base = torch.randn(m, 2 * h + 8, device="cuda", dtype=dtype)
    x = base[:, : 2 * h]
    out = torch.empty(m, h, device="cuda", dtype=dtype)
    original = x.clone()
    actual = apply_moe_activation(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        out,
        x,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
        sigmoid_mode="division",
    )
    assert actual is out and torch.equal(x, original)
    if m:
        xc = x.cpu().float()
        g = xc[:, :h].clamp(max=7)
        u = xc[:, h:].clamp(-7, 7)
        assert_fp32(out, g / (1 + torch.exp(-1.702 * g)) * (u + 1))


def test_activation_alias_and_mode_checks():
    x = torch.randn(2, 384, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        apply_moe_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            x[:, :192],
            x,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
            sigmoid_mode="division",
        )
    with pytest.raises(ValueError):
        apply_moe_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            torch.empty_like(x[:, :192]),
            x,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
            sigmoid_mode="unknown",
        )


def test_initial_baseline_cases():
    from functools import partial

    import torch
    from flaggems_vllm import fused_minimax_m3_qknorm_rope_kv_insert

    op = partial(fused_minimax_m3_qknorm_rope_kv_insert, numerics="bf16_staged")

    torch.manual_seed(704)
    torch.set_num_threads(2)
    t = torch.arange(4096).float()
    freq = 1 / (5000000.0 ** (torch.arange(0, 64, 2).float() / 64))
    cs = torch.cat(
        ((t[:, None] * freq).cos(), (t[:, None] * freq).sin()), -1
    ).bfloat16()

    def ref(x, w, pos):
        f = x.float()
        z = (
            (
                f
                * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)
                * (1 + w.float())
            )
            .bfloat16()
            .float()
        )
        out = z.clone()
        c = cs[pos, :32].float()[:, None, :]
        s = cs[pos, 32:].float()[:, None, :]
        out[:, :, :32] = z[:, :, :32] * c - z[:, :, 32:64] * s
        out[:, :, 32:64] = z[:, :, 32:64] * c + z[:, :, :32] * s
        return out.bfloat16()

    results = []
    for m in (1, 7, 129):
        for nq, nk in ((4, 1), (8, 2)):
            for sparse in (False, True):
                ni = 1 if sparse else 0
                qs = nq * 128
                ks = nk * 128
                isz = ni * 128
                original = torch.randn(
                    m, qs + 2 * ks + (isz + 128 if sparse else 0)
                ).bfloat16()
                w = [(torch.randn(128) * 0.2).bfloat16() for _ in range(4)]
                pos = (torch.arange(m) * 31 % 4096).long()
                a = original.cuda()
                wg = [v.cuda() for v in w]
                cg = cs.cuda()
                pg = pos.cuda()
                qr = ref(original[:, :qs].view(m, nq, 128), w[0], pos).view(m, qs)
                kr = ref(original[:, qs : qs + ks].view(m, nk, 128), w[1], pos).view(
                    m, ks
                )
                v = original[:, qs + ks : qs + 2 * ks]
                if not sparse:
                    op(a, wg[0], wg[1], cg, pg, nq, nk, 64, 1e-6)
                    qo = a[:, :qs]
                else:
                    cap = 1024
                    slots = (torch.arange(m) * 17) % cap
                    islots = (slots * 7 + 13) % cap
                    invalid = torch.arange(m) % 11 == 0
                    slots[invalid] = -1
                    islots[invalid] = -1
                    cache = torch.full(
                        (8, 2, 128, nk, 128), -77.0, device="cuda", dtype=torch.bfloat16
                    )
                    icache = torch.full(
                        (8, 128, 128), -77.0, device="cuda", dtype=torch.bfloat16
                    )
                    qo = torch.empty(m, qs, device="cuda", dtype=torch.bfloat16)
                    iqo = torch.empty(m, 128, device="cuda", dtype=torch.bfloat16)
                    op(
                        a,
                        wg[0],
                        wg[1],
                        cg,
                        pg,
                        nq,
                        nk,
                        64,
                        1e-6,
                        wg[2],
                        wg[3],
                        ni,
                        slots.cuda(),
                        islots.cuda(),
                        cache,
                        icache,
                        128,
                        qo,
                        iqo,
                        "bfloat16",
                    )
                    iqr = ref(
                        original[:, qs + 2 * ks : qs + 2 * ks + 128].view(m, 1, 128),
                        w[2],
                        pos,
                    ).view(m, 128)
                    ikr = ref(original[:, -128:].view(m, 1, 128), w[3], pos).view(
                        m, 128
                    )
                    torch.testing.assert_close(iqo.cpu(), iqr, rtol=0.01, atol=0.01)
                    torch.testing.assert_close(
                        a[:, -128:].cpu(), ikr, rtol=0.01, atol=0.01
                    )
                    assert torch.equal(a[:, :qs].cpu(), original[:, :qs])
                    expected = torch.full(
                        cache.shape, -77.0, device="cpu", dtype=torch.bfloat16
                    )
                    iexpected = torch.full(
                        icache.shape, -77.0, device="cpu", dtype=torch.bfloat16
                    )
                    for i in range(m):
                        if slots[i] >= 0:
                            sl = int(slots[i])
                            expected[sl // 128, 0, sl % 128] = kr[i].view(nk, 128)
                            expected[sl // 128, 1, sl % 128] = v[i].view(nk, 128)
                            si = int(islots[i])
                            iexpected[si // 128, si % 128] = ikr[i]
                    torch.testing.assert_close(
                        cache.cpu(), expected, rtol=0.01, atol=0.01
                    )
                    torch.testing.assert_close(
                        icache.cpu(), iexpected, rtol=0.01, atol=0.01
                    )
                    assert torch.equal(cache[:, 1].cpu(), expected[:, 1])
                torch.testing.assert_close(qo.cpu(), qr, rtol=0.01, atol=0.01)
                torch.testing.assert_close(
                    a[:, qs : qs + ks].cpu(), kr, rtol=0.01, atol=0.01
                )
                assert torch.equal(a[:, qs + ks : qs + 2 * ks].cpu(), v)
                results.append(
                    {
                        "tokens": m,
                        "q_heads": nq,
                        "kv_heads": nk,
                        "sparse": sparse,
                        "q_max_abs": (qo.cpu().float() - qr.float()).abs().max().item(),
                    }
                )
                print("QKV_CASE_PASS", results[-1], flush=True)
    print("QKV_PASS", len(results), flush=True)


@pytest.mark.parametrize("m,n", [(1, 128), (64, 6144), (128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_moe_sum_order(m, n, dtype):
    inp = torch.zeros(m, 4, n, device="cuda", dtype=dtype)
    inp[:, 0] = 1e20
    inp[:, 1] = 1
    inp[:, 2] = -1e20
    inp[:, 3] = 1
    out = torch.empty(m, n, device="cuda", dtype=dtype)
    fgv.moe_sum(inp, out, reduction="sequential")
    ref = torch.zeros(m, n, dtype=torch.float32)
    for i in range(4):
        ref += inp[:, i].cpu().float()
    assert torch.equal(out.cpu(), ref.to(dtype))


@pytest.mark.parametrize("kind", ["gemma", "swiglu", "qkv", "moe_sum"])
def test_numerical_mode_changed_input_graph(kind):
    torch.manual_seed(561)
    x = torch.randn(7, 768, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(768, device="cuda", dtype=x.dtype)
    norm_weight = torch.randn(128, device="cuda", dtype=x.dtype)
    cos_sin = torch.randn(128, 64, device="cuda", dtype=x.dtype)
    positions = torch.arange(7, device="cuda", dtype=torch.int64)

    def operation(a, r):
        if kind == "gemma":
            return fgv.fused_add_rms_norm(a, r, (768,), weight, 1e-6, gemma=True)
        if kind == "swiglu":
            out = torch.empty(7, 384, device=a.device, dtype=a.dtype)
            return (
                apply_moe_activation(
                    MoEActivation.SWIGLUOAI_UNINTERLEAVE,
                    out,
                    a,
                    gemm1_alpha=1.702,
                    gemm1_beta=1.0,
                    gemm1_clamp_limit=7.0,
                    sigmoid_mode="division",
                ),
            )
        if kind == "qkv":
            fgv.fused_minimax_m3_qknorm_rope_kv_insert(
                a,
                norm_weight,
                norm_weight,
                cos_sin,
                positions,
                4,
                1,
                64,
                1e-6,
                numerics="bf16_staged",
            )
            return (a,)
        out = torch.empty(7, 192, device=a.device, dtype=a.dtype)
        fgv.moe_sum(a.view(7, 4, 192), out, reduction="sequential")
        return (out,)

    for _ in range(3):
        operation(x.clone(), residual.clone())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = operation(x, residual)
    for scale in (0.25, -1.0, 2.0):
        fresh = torch.randn_like(x) * scale
        fresh_residual = torch.randn_like(residual)
        expected = operation(fresh.clone(), fresh_residual.clone())
        x.copy_(fresh)
        residual.copy_(fresh_residual)
        graph.replay()
        torch.cuda.synchronize()
        for result, reference in zip(actual, expected):
            assert torch.equal(result, reference)


@pytest.mark.parametrize("kind", ["gemma", "swiglu"])
@pytest.mark.parametrize("m", [1, 64, 8192])
def test_disjoint_shared_workspace(kind, m):
    """vLLM workspaces can be disjoint slices of one allocator-owned storage."""
    torch.manual_seed(562)
    n = 6144 if kind == "gemma" else 384
    out_n = n if kind == "gemma" else n // 2
    left_size, right_size = m * n, m * out_n
    arena = torch.empty(
        left_size + 64 + right_size, device="cuda", dtype=torch.bfloat16
    )
    x = arena[:left_size].view(m, n)
    other = arena[left_size + 64 :].view(m, out_n)
    guard = arena[left_size : left_size + 64]
    guard.fill_(1.625)
    weight = torch.randn(n, device="cuda", dtype=x.dtype) * 0.2

    def reset():
        x.normal_()
        other.normal_()
        a, b = x.cpu().float(), other.cpu().float()
        if kind == "gemma":
            added = a + b
            expected = added * torch.rsqrt(added.square().mean(-1, keepdim=True) + 1e-6)
            return expected * (1 + weight.cpu().float()), added.bfloat16()
        gate, up = a.chunk(2, -1)
        gate, up = gate.clamp(max=7), up.clamp(-7, 7)
        return gate / (1 + torch.exp(-1.702 * gate)) * (up + 1), None

    def operation():
        if kind == "gemma":
            fgv.fused_add_rms_norm(x, other, (n,), weight, 1e-6, gemma=True)
        else:
            apply_moe_activation(
                MoEActivation.SWIGLUOAI_UNINTERLEAVE,
                other,
                x,
                gemm1_alpha=1.702,
                gemm1_beta=1.0,
                gemm1_clamp_limit=7.0,
                sigmoid_mode="division",
            )

    def check(reference):
        expected, residual = reference
        assert_fp32(x if kind == "gemma" else other, expected)
        if residual is not None:
            assert torch.equal(other.cpu(), residual)
        assert torch.equal(guard, torch.full_like(guard, 1.625))

    assert x.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()
    expected = reset()
    operation()
    check(expected)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    expected = reset()
    graph.replay()
    torch.cuda.synchronize()
    check(expected)
