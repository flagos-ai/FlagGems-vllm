# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm as fgv

pytestmark = [
    pytest.mark.skipif(fgv.vendor_name != "metax", reason="MetaX backend"),
]


@pytest.mark.parametrize(
    "m,k,width,start,hidden",
    [
        (1, 1, 1, 0, 1),
        (2, 3, 255, 2, 263),
        (12, 7, 257, 1, 270),
        (64, 16, 3584, 0, 3584),
        (5, 16, 1024, 1024, 3584),
        (3, 16, 512, 3072, 3584),
    ],
)
@pytest.mark.moe_weighted_sum_out
def test_weighted_sum_matches_exact_reference_after_capture(m, k, width, start, hidden):
    values = torch.empty((m * k, width), device="cuda", dtype=torch.bfloat16)
    weights = torch.empty((m, k), device="cuda", dtype=torch.float32)
    out = torch.empty((m, hidden), device="cuda", dtype=torch.bfloat16)
    apply = fgv.moe_weighted_sum_out
    values.zero_()
    weights.zero_()
    apply(values, weights, out, column_start=start)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply(values, weights, out, column_start=start)

    rng = torch.Generator().manual_seed(803)
    for _ in range(3):
        a = torch.randint(-8, 9, (m, k, width), generator=rng)
        b = torch.randint(-6, 7, (m, k), generator=rng)
        values.copy_(a.reshape(m * k, width).float() / 8)
        weights.copy_(b.float() / 16)
        out.fill_(float("nan"))
        graph.replay()
        actual = out.cpu()
        expected = ((a * b[:, :, None]).sum(1).double() / 128).bfloat16()
        torch.testing.assert_close(
            actual[:, start : start + width], expected, rtol=0, atol=0
        )
        assert (
            actual[:, :start].isnan().all() and actual[:, start + width :].isnan().all()
        )
        assert torch.equal(values.cpu().float(), a.view(m * k, width).float() / 8)
        assert torch.equal(weights.cpu(), b.float() / 16)


@pytest.mark.moe_weighted_sum_out
def test_weighted_sum_rounds_only_after_reduction():
    values = torch.tensor([[1], [-1]], device="cuda", dtype=torch.bfloat16)
    weights = torch.tensor([[1.0009765625, 1.0]], device="cuda")
    out = torch.empty_like(values[:1])
    fgv.moe_weighted_sum_out(values, weights, out)
    assert out.item() == 2**-10


@pytest.mark.moe_weighted_sum_out
def test_output_storage_and_empty_contract():
    values = torch.ones((4, 4), device="cuda", dtype=torch.bfloat16)
    weights = torch.ones((2, 2), device="cuda")
    out = torch.full((2, 4), -7, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="alias"):
        fgv.moe_weighted_sum_out(values, weights, values[:2])
    with pytest.raises(ValueError, match="slice"):
        fgv.moe_weighted_sum_out(values, weights, out, column_start=1)
    with pytest.raises(TypeError, match="column_start"):
        fgv.moe_weighted_sum_out(values, weights, out, column_start=0.5)
    with pytest.raises(ValueError, match="FP32"):
        fgv.moe_weighted_sum_out(values, weights.bfloat16(), out)
    with pytest.raises(ValueError, match="contiguous"):
        fgv.moe_weighted_sum_out(values.T, weights, out)
    assert out.eq(-7).all()
    assert fgv.moe_weighted_sum_out(values[:0], weights[:0], out[:0]) is None
