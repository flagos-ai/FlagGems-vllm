# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm
from baseline.megagate import (
    GATE_PROFILES,
    generate_gate_inputs,
    router_scores,
    torch_megagate,
)

from . import conftest as cfg

pytestmark = [
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "thead" or not torch.cuda.is_available(),
        reason="requires T-Head PPU",
    ),
]


def _assert_routes(result, expected):
    weights, ids = result
    ref_weights, ref_ids = expected
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int64
    # The native router may order equal scores differently. Compare by ID.
    ids, order = ids.sort(dim=-1)
    ref_ids, ref_order = ref_ids.sort(dim=-1)
    torch.testing.assert_close(ids, ref_ids)
    torch.testing.assert_close(
        weights.gather(-1, order),
        ref_weights.gather(-1, ref_order),
        rtol=1e-3,
        atol=1e-4,
    )


@pytest.mark.megagate
@pytest.mark.parametrize("name", GATE_PROFILES)
@pytest.mark.parametrize("tokens", [1, 33] if cfg.QUICK_MODE else [1, 33, 1025])
def test_megagate_projection_and_routing(name, tokens):
    args, kwargs = generate_gate_inputs(GATE_PROFILES[name], tokens, "cuda")
    x, weight, k = args
    # Compute the reference projection with FP64 on CPU independently of GEMM.
    x_cpu, w_cpu = x.cpu().double(), weight.cpu().double()
    precise_logits = x_cpu @ w_cpu.T
    logits = precise_logits.to(kwargs["router_logits_dtype"])
    scores = router_scores(logits, kwargs["scoring_func"])
    bias = kwargs["bias"]
    ranking = scores if bias is None else scores + bias.cpu()
    groups, top_groups = kwargs["num_expert_group"], kwargs["topk_group"]
    if groups > top_groups:
        grouped = ranking.unflatten(-1, (groups, -1))
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        group_ids = group_scores.topk(top_groups, dim=-1).indices
        mask = torch.zeros_like(group_scores, dtype=torch.bool)
        mask.scatter_(1, group_ids, True)
        ranking = grouped.masked_fill(~mask.unsqueeze(-1), -torch.inf).flatten(1)
    # The kernel's softmax selection exploits the monotonicity of exp.
    if kwargs["scoring_func"] == "softmax":
        ranking = logits.float()
    ref_ids = ranking.argsort(dim=-1, descending=True, stable=True)[:, :k]
    result = flaggems_vllm.megagate(*args, **kwargs)
    ids = result[1].cpu()
    assert ((0 <= ids) & (ids < weight.shape[0])).all()
    sorted_ids = ids.sort(dim=-1).values
    assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()
    # Selection can differ around a rounding boundary. The two sets must
    # have the same ranking scores within the gate projection tolerance.
    if kwargs["router_logits_dtype"] == torch.bfloat16:
        torch.testing.assert_close(
            ranking.gather(-1, ids).sort(dim=-1).values,
            ranking.gather(-1, ref_ids).sort(dim=-1).values,
            rtol=9e-3,
            atol=1e-5,
        )
    else:
        torch.testing.assert_close(
            ranking.gather(-1, ids).sort(dim=-1).values,
            ranking.gather(-1, ref_ids).sort(dim=-1).values,
            rtol=2e-5,
            atol=2e-5,
        )
    if kwargs["router_logits_dtype"] == torch.bfloat16:
        # FP32 accumulation can put a result on either side of a BF16
        # midpoint. Bound each selected score using both rounded endpoints.
        tolerance = 2e-5
        low = (precise_logits - tolerance).bfloat16().float().gather(-1, ids)
        high = (precise_logits + tolerance).bfloat16().float().gather(-1, ids)
        if kwargs["scoring_func"] == "softmax":
            shift = high.amax(-1, keepdim=True)
            low, high = (low - shift).exp(), (high - shift).exp()
        else:
            low = router_scores(low, kwargs["scoring_func"])
            high = router_scores(high, kwargs["scoring_func"])
        scale = kwargs["routed_scaling_factor"]
        min_weight = (low / (high.sum(-1, keepdim=True) - high + low)) * scale
        max_weight = (high / (low.sum(-1, keepdim=True) - low + high)) * scale
        actual = result[0].cpu()
        nearest = actual.clamp(min=min_weight, max=max_weight)
        torch.testing.assert_close(actual, nearest, rtol=1e-3, atol=1e-4)
    else:
        values = scores.gather(-1, ids)
        values = values / values.sum(-1, keepdim=True) * kwargs["routed_scaling_factor"]
        torch.testing.assert_close(result[0].cpu(), values, rtol=1e-3, atol=1e-4)


@pytest.mark.megagate
def test_megagate_hash_image_and_unbiased_scores():
    x = torch.ones((4, 257), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((7, 257), dtype=torch.bfloat16, device="cuda")
    weight[:, 0] = torch.arange(7, device="cuda")
    fixed = torch.tensor(
        [[4, 2, 3, -1], [6, 0, 2, -1], [6, 0, 2, -1], [6, 0, 2, -1]],
        device="cuda",
    )[:, :3]
    image_mask = torch.tensor([False, True, False, True], device="cuda")
    fixed_mask = torch.tensor([True, False, False, True], device="cuda")
    bias = torch.tensor([20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0], device="cuda")
    image_bias = torch.tensor([0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 0.0], device="cuda")
    expected_ids = torch.tensor(
        [[4, 2, 3], [5, 4, 3], [2, 1, 0], [6, 0, 2]], device="cuda"
    )
    weights, ids = flaggems_vllm.megagate(
        x,
        weight,
        3,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_mask,
        unmapped_topk_idx=fixed,
        fix_routing_mask=fixed_mask,
        routed_scaling_factor=1.5,
    )
    torch.testing.assert_close(ids, expected_ids)
    scores = router_scores(expected_ids, "sqrtsoftplus")
    expect = scores / scores.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(weights, expect, rtol=1e-4, atol=1e-5)


@pytest.mark.megagate
@pytest.mark.parametrize("scoring", ["sqrtsoftplus", "sigmoid", "identity", "softmax"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_megagate_scoring_semantics(scoring, renormalize):
    x = torch.ones((3, 16), device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros((23, 16), device="cuda", dtype=torch.bfloat16)
    weight[:, 0] = torch.linspace(-40, 20, 23, device="cuda")
    args = (x, weight, 5)
    kwargs = {"scoring_func": scoring, "renormalize": renormalize}
    _assert_routes(
        flaggems_vllm.megagate(*args, **kwargs), torch_megagate(*args, **kwargs)
    )


@pytest.mark.megagate
def test_megagate_empty_and_ties():
    x = torch.zeros((0, 32), device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros((29, 32), device="cuda", dtype=torch.bfloat16)
    weights, ids = flaggems_vllm.megagate(x, weight, 5)
    assert weights.shape == ids.shape == (0, 5)
    x = torch.zeros((3, 32), device="cuda", dtype=torch.bfloat16)
    weights, ids = flaggems_vllm.megagate(x, weight, 5)
    expected = torch.arange(5, device="cuda").expand_as(ids)
    torch.testing.assert_close(ids, expected)
    torch.testing.assert_close(weights, torch.full_like(weights, 0.2))


@pytest.mark.megagate
def test_megagate_invalid_inputs():
    x = torch.zeros((2, 32), device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros((16, 32), device="cuda", dtype=torch.bfloat16)
    op = flaggems_vllm.megagate
    with pytest.raises(ValueError, match="num_topk"):
        op(x, weight, 0)
    with pytest.raises(ValueError, match="contiguous"):
        op(x.T, weight, 4)
    with pytest.raises(NotImplementedError, match="BF16"):
        op(x.float(), weight, 4)
    with pytest.raises(ValueError, match="supplied together"):
        op(x, weight, 4, image_bias=torch.empty(16, device="cuda"))
    with pytest.raises(ValueError, match="requires unmapped"):
        op(
            x,
            weight,
            4,
            fix_routing_mask=torch.ones(2, device="cuda", dtype=torch.bool),
        )


@pytest.mark.megagate
def test_megagate_concurrent_graph_replay():
    profile = GATE_PROFILES["dsv41-flash"]
    runs = []
    for tokens in (8, 33):
        args, kwargs = generate_gate_inputs(profile, tokens, "cuda")
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        flaggems_vllm.megagate(*args, **kwargs)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            out = flaggems_vllm.megagate(*args, **kwargs)
        runs.append((args, kwargs, stream, graph, out))

    for _ in range(3):
        for args, kwargs, stream, graph, _ in runs:
            with torch.cuda.stream(stream):
                args[0].neg_()
                kwargs["bias"].copy_(kwargs["bias"].roll(1))
                graph.replay()
        torch.cuda.synchronize()
        for args, kwargs, _, _, out in runs:
            _assert_routes(out, torch_megagate(*args, **kwargs))


@pytest.mark.megagate
def test_megagate_fixed_rows_and_nonfinite_padding():
    x = torch.ones((3, 32), device="cuda", dtype=torch.bfloat16)
    x[1].fill_(float("nan"))
    x[2].fill_(float("inf"))
    weight = torch.ones((16, 32), device="cuda", dtype=torch.bfloat16)
    fixed = torch.tensor([[1, 3, 2**32 + 8], [4, 5, 6], [9, 10, 11]], device="cuda")
    weights, ids = flaggems_vllm.megagate(x, weight, 3, unmapped_topk_idx=fixed)
    expected = fixed.clone()
    expected[0, 2] = -1
    torch.testing.assert_close(ids, expected)
    torch.testing.assert_close(weights[0], weights.new_tensor([0.5, 0.5, 0.0]))
    assert weights[1:].isnan().all()


@pytest.mark.megagate
def test_megagate_portable_projection(monkeypatch):
    from importlib import import_module

    impl = import_module("flaggems_vllm.runtime.backend._thead.ops.megagate")
    monkeypatch.setattr(impl, "_HAS_ASYNC_LOAD", False)
    monkeypatch.setattr(impl, "_load_memset", lambda: None)
    args, kwargs = generate_gate_inputs(GATE_PROFILES["dsv4-flash"], 7, "cuda")
    _assert_routes(
        flaggems_vllm.megagate(*args, **kwargs), torch_megagate(*args, **kwargs)
    )
