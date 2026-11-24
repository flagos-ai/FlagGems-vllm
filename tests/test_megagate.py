# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MegaGate correctness tests."""

import ctypes
from dataclasses import dataclass
from functools import partial

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm

HYGON_MEGAGATE_AVAILABLE = (
    flaggems_vllm.vendor_name == "hygon"
    and torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx936")
)


@dataclass(frozen=True)
class HygonMegaGateProfile:
    name: str
    hidden: int
    experts: int
    topk: int
    scale: float
    routing: str
    tokens: tuple[int, ...]


# Exact supported shapes in FlagGems-vllm PR #1196 at ac152112.
HYGON_PROFILES = (
    HygonMegaGateProfile(
        "dsv4_flash",
        4096,
        256,
        6,
        1.5,
        "sqrtsoftplus",
        (1028, 2048, 3076, 3805, 4100, 4835, 7884, 8221, 13021, 16384),
    ),
    HygonMegaGateProfile(
        "dsv32",
        7168,
        256,
        8,
        2.5,
        "grouped_sigmoid",
        (1028, 1756, 2048, 3076, 3805, 8276, 12292, 12372, 14305, 16384),
    ),
    HygonMegaGateProfile(
        "qwen_h2048_e256_k8",
        2048,
        256,
        8,
        1.0,
        "softmax",
        (
            1034,
            1035,
            1036,
            1352,
            2048,
            3104,
            4107,
            4108,
            4138,
            4434,
            4435,
            4436,
            9309,
            9330,
            11469,
            13421,
            13422,
            16384,
        ),
    ),
    HygonMegaGateProfile(
        "qwen_h4096_e512_k10",
        4096,
        512,
        10,
        1.0,
        "softmax",
        (1035, 1036, 2048, 4107, 4108, 4434, 4435, 5240, 8214, 13421, 13422, 16384),
    ),
)


HYGON_CASES = tuple(
    (profile, tokens) for profile in HYGON_PROFILES for tokens in profile.tokens
)


def prepare_hygon_megagate(profile, x, weight, bias):
    from flaggems_vllm.runtime.backend._hygon.ops.megagate import MegaGateContext

    context = MegaGateContext(profile.name)
    context.prepare(x)
    return partial(
        flaggems_vllm.megagate, x, weight, bias, profile=profile.name, context=context
    )


def make_hygon_inputs(profile, tokens):
    generator = torch.Generator(device="cuda").manual_seed(20261112 + tokens)
    x = torch.randn(
        (tokens, profile.hidden),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        (profile.experts, profile.hidden),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(profile.hidden**-0.5)
    bias = None
    if profile.routing != "softmax":
        bias = torch.randn(
            (profile.experts,), device="cuda", dtype=torch.float32, generator=generator
        ).mul_(0.01)
    return x, weight, bias


def _hygon_project(profile, x, weight):
    if profile.routing == "sqrtsoftplus":
        return torch.mm(x, weight.T, out_dtype=torch.float32)
    return torch.mm(x, weight.T)


def capture_hygon_graph(run, calls):
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        for _ in range(calls):
            result = run()
    graph.replay()
    return graph, result


def _count_hygon_kernel_nodes(graph):
    hip = ctypes.CDLL("libgalaxyhip.so.5")
    handle = ctypes.c_void_p(graph.raw_cuda_graph())
    node_count = ctypes.c_size_t()
    byref = ctypes.byref
    if hip.hipGraphGetNodes(handle, None, byref(node_count)):
        raise RuntimeError("hipGraphGetNodes failed")
    nodes = (ctypes.c_void_p * node_count.value)()
    if hip.hipGraphGetNodes(handle, nodes, byref(node_count)):
        raise RuntimeError("hipGraphGetNodes failed")
    kernels = 0
    for node in nodes:
        kind = ctypes.c_int()
        if hip.hipGraphNodeGetType(ctypes.c_void_p(node), byref(kind)):
            raise RuntimeError("hipGraphNodeGetType failed")
        # The DTK runtime reports native kernel nodes as type 200.
        if kind.value == 4:
            raise RuntimeError("Nested graph nodes require separate launch accounting")
        kernels += kind.value in (0, 200)
    return kernels


def _check_hygon_routing(profile, logits, bias, result):
    weights, indices = result
    expected_shape = (logits.shape[0], profile.topk)
    index_dtype = torch.int64 if profile.routing == "sqrtsoftplus" else torch.int32
    assert weights.shape == indices.shape == expected_shape
    assert weights.dtype == torch.float32 and indices.dtype == index_dtype
    ids = indices.long()
    sorted_ids = ids.sort(dim=-1).values
    assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()

    logits = logits.float()
    if profile.routing == "softmax":
        # Renormalizing the selected full-softmax weights is equivalent to this.
        ranking = logits
        expected_weights = logits.gather(1, ids).softmax(dim=-1)
    else:
        scores = (
            F.softplus(logits).sqrt()
            if profile.routing == "sqrtsoftplus"
            else logits.sigmoid()
        )
        ranking = scores + bias
        if profile.routing == "grouped_sigmoid":
            grouped = ranking.view(-1, 8, 32)
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
            group_ids = group_scores.topk(4, dim=-1).indices
            mask = torch.zeros_like(group_scores, dtype=torch.bool)
            mask.scatter_(1, group_ids, True)
            ranking = grouped.masked_fill(~mask.unsqueeze(-1), -torch.inf).flatten(1)
        selected = scores.gather(1, ids)
        expected_weights = selected / selected.sum(dim=-1, keepdim=True) * profile.scale

    # Compare ranking values to allow equivalent expert ids at exact score ties.
    selected_ranks = ranking.gather(1, ids).sort(dim=-1, descending=True).values
    expected_ranks = ranking.topk(profile.topk, dim=-1).values
    torch.testing.assert_close(selected_ranks, expected_ranks, rtol=0, atol=2e-6)
    torch.testing.assert_close(weights, expected_weights, rtol=5e-3, atol=5e-4)


def _check_bf16_softmax_boundary(exact_logits, result):
    weights, indices = result
    assert weights.shape == indices.shape and indices.dtype == torch.int32
    assert weights.dtype == torch.float32
    ids = indices.long()
    ordered = ids.sort(dim=-1).values
    assert (ordered[:, 1:] != ordered[:, :-1]).all()

    # Bound FP32 accumulation error before rounding each logit to BF16.
    # Only values within this interval of a BF16 midpoint can round differently.
    lower = (exact_logits - 1e-5).to(torch.bfloat16).float()
    upper = (exact_logits + 1e-5).to(torch.bfloat16).float()
    low_selected = lower.gather(1, ids)
    high_selected = upper.gather(1, ids)
    max_excluded = lower.scatter(1, ids, -torch.inf).amax(dim=1)
    assert (high_selected.amin(dim=1) >= max_excluded).all()

    low_exp = low_selected.exp()
    high_exp = high_selected.exp()
    weight_min = low_exp / (low_exp + high_exp.sum(dim=1, keepdim=True) - high_exp)
    weight_max = high_exp / (high_exp + low_exp.sum(dim=1, keepdim=True) - low_exp)
    assert ((weights >= weight_min - 5e-5) & (weights <= weight_max + 5e-5)).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones_like(max_excluded))


def _check_hygon_megagate(profile, logits, x, weight, bias, result):
    assert result[0].shape == result[1].shape == (x.shape[0], profile.topk)
    try:
        _check_hygon_routing(profile, logits, bias, result)
    except AssertionError:
        if logits.dtype != torch.bfloat16:
            raise
        # Independent wide accumulation adjudicates BF16 rounding boundaries.
        exact_logits = x.double() @ weight.T.double()
        if profile.routing == "softmax":
            _check_bf16_softmax_boundary(exact_logits, result)
        else:
            try:
                _check_hygon_routing(
                    profile, exact_logits.to(torch.bfloat16), bias, result
                )
            except AssertionError:
                # Require a complete grouped routing match to an attainable
                # BF16 projection on either side of an FP32 rounding boundary.
                for offset in (-1e-5, 1e-5):
                    rounded = (exact_logits + offset).to(torch.bfloat16)
                    try:
                        _check_hygon_routing(profile, rounded, bias, result)
                    except AssertionError:
                        continue
                    return True
                raise


@pytest.mark.megagate
@pytest.mark.skipif(not HYGON_MEGAGATE_AVAILABLE, reason="Requires Hygon gfx936")
@pytest.mark.parametrize(
    "profile,tokens",
    [pytest.param(p, t, id=f"{p.name}-T{t}") for p, t in HYGON_CASES],
)
@torch.inference_mode()
def test_megagate_hygon(profile, tokens):
    x, weight, bias = make_hygon_inputs(profile, tokens)
    run = prepare_hygon_megagate(profile, x, weight, bias)
    graph, result = capture_hygon_graph(run, 1)
    assert _count_hygon_kernel_nodes(graph) == 1
    _check_hygon_megagate(
        profile, _hygon_project(profile, x, weight), x, weight, bias, result
    )

    # Changing the input between replays checks reuse of the prepared workspace.
    x.neg_()
    graph.replay()
    _check_hygon_megagate(
        profile, _hygon_project(profile, x, weight), x, weight, bias, result
    )
