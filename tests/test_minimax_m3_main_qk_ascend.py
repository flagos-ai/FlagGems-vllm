# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.fused_minimax_m3_qknorm_rope_kv_insert,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="requires Ascend"),
]


def reference(x, weight, phase, rotary_dim):
    x = x.double()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    x = x * (1 + weight.double())
    cos, sin = phase.double().unsqueeze(1).chunk(2, -1)
    a, b = x[..., :rotary_dim].chunk(2, -1)
    return torch.cat(
        (a * cos - b * sin, b * cos + a * sin, x[..., rotary_dim:]), -1
    ).bfloat16()


@pytest.mark.parametrize("rotary_dim", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("gather", [False, True])
def test_main_qk(rotary_dim, gather):
    gen = torch.Generator().manual_seed(53)

    def rand(*shape):
        return torch.randn(shape, generator=gen, dtype=torch.bfloat16)

    qkv = rand(7, 8, 128)
    qw, kw, cache = rand(128), rand(128), rand(19, rotary_dim)
    pos = torch.tensor([11, 2, 18, 0, 2, 9, 6])
    # A sparse layer appends index Q/K to each projected row.
    storage = torch.full((7, 10 * 128), 17, dtype=qkv.dtype, device="npu")
    packed = storage[:, : 8 * 128] if gather else qkv.flatten(1).to("npu")
    packed.copy_(qkv.flatten(1))
    out = (
        torch.full((7, 4, 128), float("nan"), dtype=qkv.dtype, device="npu")
        if gather
        else None
    )
    assert (
        flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
            packed,
            qw.npu(),
            kw.npu(),
            cache.npu(),
            pos.npu(),
            4,
            2,
            rotary_dim,
            1e-6,
            q_out=out,
        )
        is None
    )
    got = packed.cpu().view_as(qkv)
    torch.testing.assert_close(
        out.cpu() if gather else got[:, :4],
        reference(qkv[:, :4], qw, cache[pos], rotary_dim),
        atol=0.016,
        rtol=0.008,
    )
    torch.testing.assert_close(
        got[:, 4:6],
        reference(qkv[:, 4:6], kw, cache[pos], rotary_dim),
        atol=0.016,
        rtol=0.008,
    )
    assert torch.equal(got[:, 6:], qkv[:, 6:])
    if gather:
        assert torch.equal(got[:, :4], qkv[:, :4])
        assert (storage[:, 8 * 128 :] == 17).all().item()


def test_replay_main_qk():
    packed = torch.zeros(3, 6 * 128, dtype=torch.bfloat16, device="npu")
    out = torch.empty_like(packed[:, :512]).contiguous()
    qw = torch.zeros(128, dtype=packed.dtype, device="npu")
    kw = torch.full_like(qw, -0.25)
    phase = torch.randn(9, 64, dtype=packed.dtype, device="npu")
    positions = torch.zeros(3, dtype=torch.int64, device="npu")

    def run(**kwargs):
        return flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
            packed, qw, kw, phase, positions, 4, 1, 64, 1e-6, q_out=out, **kwargs
        )

    with pytest.raises(NotImplementedError):
        run(num_index_heads=1)
    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for pos in ([1, 7, 2], [8, 0, 3]):
        original = torch.randn(3, 6, 128, dtype=packed.dtype)
        packed.copy_(original.flatten(1))
        positions.copy_(torch.tensor(pos))
        out.fill_(float("nan"))
        graph.replay()
        expected_q = reference(original[:, :4], qw.cpu(), phase.cpu()[pos], 64)
        expected_k = reference(original[:, 4:5], kw.cpu(), phase.cpu()[pos], 64)
        got = packed.cpu().view_as(original)
        torch.testing.assert_close(
            out.cpu().view_as(expected_q), expected_q, atol=0.016, rtol=0.008
        )
        torch.testing.assert_close(got[:, 4:5], expected_k, atol=0.016, rtol=0.008)
        assert torch.equal(got[:, :4], original[:, :4])
        assert torch.equal(got[:, 5:], original[:, 5:])
