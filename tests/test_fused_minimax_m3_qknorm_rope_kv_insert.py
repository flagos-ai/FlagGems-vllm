# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.fused_minimax_m3_qknorm_rope_kv_insert,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def literal_norm_rope(x, weight, cs, positions, rotary):
    x = x.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight.float())
    phase = cs[positions].float()
    half = rotary // 2
    a, b = y[..., :half], y[..., half:rotary]
    c, s = phase[:, :half, None].transpose(1, 2), phase[:, half:, None].transpose(1, 2)
    return torch.cat((a * c - b * s, b * c + a * s, y[..., rotary:]), dim=-1)


@pytest.mark.parametrize("tokens", [0, 1, 7, 32])
@pytest.mark.parametrize("rotary", [8, 64, 128])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("gather", [False, True])
def test_fused_minimax_m3_cache(tokens, rotary, fp8, gather):
    torch.manual_seed(37)
    nq, nkv, niq = 4, 2, 2
    heads = nq + 2 * nkv + niq + 1
    qkv = torch.randn((tokens, heads * 128), device="cuda", dtype=torch.bfloat16)
    original = qkv.view(tokens, heads, 128).clone()
    weights = [
        torch.randn((128,), device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(4)
    ]
    cs = torch.randn((128, rotary), device="cuda", dtype=torch.bfloat16) * 0.25
    pos = torch.arange(tokens, device="cuda", dtype=torch.int64)
    slots = pos.clone()
    index_slots = pos.flip(0).contiguous()
    if tokens > 1:
        slots[1] = -1
        index_slots[1] = -1
    cache = torch.full(
        (4, 2, 16, nkv, 128),
        11,
        device="cuda",
        dtype=torch.uint8 if fp8 else torch.bfloat16,
    )
    index = torch.full((64, 128), 11, device="cuda", dtype=torch.bfloat16)
    qout = (
        torch.empty((tokens, nq, 128), device="cuda", dtype=torch.bfloat16)
        if gather
        else None
    )
    iqout = (
        torch.empty((tokens, niq, 128), device="cuda", dtype=torch.bfloat16)
        if gather
        else None
    )
    flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        *weights[:2],
        cs,
        pos,
        nq,
        nkv,
        rotary,
        1e-6,
        *weights[2:],
        niq,
        slots,
        index_slots,
        cache,
        index,
        16,
        qout,
        iqout,
        "fp8_e4m3" if fp8 else "auto",
    )
    expected = original.float().clone()
    for start, end, w in [
        (0, nq, weights[0]),
        (nq, nq + nkv, weights[1]),
        (nq + 2 * nkv, nq + 2 * nkv + niq, weights[2]),
        (heads - 1, heads, weights[3]),
    ]:
        expected[:, start:end] = literal_norm_rope(
            original[:, start:end], w, cs, pos, rotary
        )
    rounded = expected.bfloat16()
    actual = qkv.view(tokens, heads, 128)
    if gather:
        torch.testing.assert_close(qout, rounded[:, :nq], atol=0.016, rtol=0.008)
        torch.testing.assert_close(
            iqout, rounded[:, nq + 2 * nkv : -1], atol=0.016, rtol=0.008
        )
        assert torch.equal(actual[:, :nq], original[:, :nq])
        assert torch.equal(actual[:, nq + 2 * nkv : -1], original[:, nq + 2 * nkv : -1])
    else:
        torch.testing.assert_close(actual, rounded, atol=0.016, rtol=0.008)
    for t, slot in enumerate(slots.cpu().tolist()):
        if slot < 0:
            continue
        want = torch.stack(
            (expected[t, nq : nq + nkv], expected[t, nq + nkv : nq + 2 * nkv])
        )
        got = cache[slot // 16, :, slot % 16]
        if fp8:
            got = got.view(torch.float8_e4m3fn).float()
            want = want.to(torch.float8_e4m3fn).float()
            torch.testing.assert_close(got, want, atol=0.125, rtol=0.125)
        else:
            torch.testing.assert_close(got, want.bfloat16(), atol=0.016, rtol=0.008)
    for t, slot in enumerate(index_slots.cpu().tolist()):
        if slot >= 0:
            torch.testing.assert_close(
                index[slot], rounded[t, -1], atol=0.016, rtol=0.008
            )
    assert bool((cache[3] == 11).all()) and bool((index[63] == 11).all())


def test_fused_minimax_m3_norm_only_graph():
    x = torch.randn((3, 8 * 128), device="cuda", dtype=torch.bfloat16)
    original = x.clone()
    w = torch.zeros((128,), device="cuda", dtype=torch.bfloat16)
    cs = torch.randn((16, 64), device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(3, device="cuda", dtype=torch.int64)
    out = torch.empty((3, 4, 128), device="cuda", dtype=torch.bfloat16)

    def run():
        flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
            x, w, w, cs, pos, 4, 2, 64, 1e-6, q_out=out
        )

    run()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    x.copy_(original)
    pos.add_(3)
    g.replay()
    expected = literal_norm_rope(
        original.view(3, 8, 128)[:, :4], w, cs, pos, 64
    ).bfloat16()
    torch.testing.assert_close(out, expected, atol=0.016, rtol=0.008)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
            x.float(), w, w, cs, pos, 4, 2, 64, 1e-6
        )


@pytest.mark.parametrize("tokens", [1, 7, 64])
@pytest.mark.parametrize("rotary", [32, 64, 128])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("hnd", [False, True])
def test_fused_minimax_m3_native_bitwise(tokens, rotary, fp8, hnd):
    pytest.importorskip("vllm._C_stable_libtorch")
    native = getattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert", None)
    if native is None:
        pytest.skip("Pinned native M3 reference is unavailable")
    torch.manual_seed(67)
    nq, nkv, niq, block = 4, 2, 2, 16
    x = torch.randn((tokens, 11 * 128), device="cuda", dtype=torch.bfloat16)
    weights = [
        torch.randn((128,), device="cuda", dtype=torch.bfloat16) * 0.3 for _ in range(4)
    ]
    angle = torch.randn((128, rotary // 2), device="cuda")
    cs = torch.cat((angle.cos(), angle.sin()), -1).bfloat16()
    pos = torch.randperm(128, device="cuda")[:tokens].long()
    slots = torch.randperm(128, device="cuda")[:tokens].long()
    slots[::7] = -1
    isl = torch.randperm(128, device="cuda")[:tokens].long()
    isl[::5] = -1
    dtype = torch.uint8 if fp8 else torch.bfloat16
    if hnd:
        cache = torch.full(
            (8, 2, nkv, block, 128), 53, device="cuda", dtype=dtype
        ).permute(0, 1, 3, 2, 4)
    else:
        cache = torch.full((8, 2, block, nkv, 128), 53, device="cuda", dtype=dtype)
    index = torch.full((8, block, 128), 31, device="cuda", dtype=torch.bfloat16)
    qout = torch.empty((tokens, nq * 128), device="cuda", dtype=torch.bfloat16)
    iqout = torch.empty((tokens, niq * 128), device="cuda", dtype=torch.bfloat16)
    args = [
        x,
        *weights[:2],
        cs,
        pos,
        nq,
        nkv,
        rotary,
        1e-6,
        *weights[2:],
        niq,
        slots,
        isl,
        cache,
        index,
        block,
        qout,
        iqout,
        "fp8_e4m3" if fp8 else "auto",
    ]
    expected = [
        (
            a.clone(memory_format=torch.preserve_format)
            if isinstance(a, torch.Tensor)
            else a
        )
        for a in args
    ]
    native(*expected)
    flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(*args)
    for i in (0, 14, 15, 17, 18):
        assert torch.equal(args[i], expected[i]), i
