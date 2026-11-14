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

import os
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter_ns

import pytest
import torch
import torch.nn.functional as F
import yaml

import flaggems_vllm

pytestmark = [
    pytest.mark.mega_gate,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "ascend", reason="requires Ascend ACL Graph"
    ),
]


@dataclass(frozen=True)
class GateCase:
    profile: str
    tokens: int
    hidden: int
    experts: int
    topk: int
    scoring: str
    groups: int = 1
    top_groups: int = 1
    scale: float = 1.0

    @property
    def id(self):
        return f"{self.profile}-t{self.tokens}"


GATE_SHAPE_FILE = Path(__file__).resolve().parents[1] / "benchmark/core_shapes.yaml"


def load_gate_cases(shape_file=GATE_SHAPE_FILE, *, quick=False):
    with open(shape_file) as stream:
        config = (yaml.safe_load(stream) or {}).get("mega_gate")
    assert config and config.get("shapes"), f"No mega_gate shapes in {shape_file}"

    cases = [GateCase(*shape) for shape in config["shapes"]]
    if quick and (quick_tokens := config.get("quick_tokens")):
        cases = [case for case in cases if case.tokens in quick_tokens]
    assert cases, f"No matching mega_gate shapes in {shape_file}"
    return cases


def gate_inputs(case, seed):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(case.tokens, case.hidden, generator=rng, dtype=torch.bfloat16)
    weight = (
        torch.randn(case.experts, case.hidden, generator=rng)
        .mul_(case.hidden**-0.5)
        .bfloat16()
    )
    bias = None
    if case.scoring != "softmax":
        bias = torch.randn(case.experts, generator=rng).mul_(0.02)
    return x, weight, bias


def gate_scores(case, x, weight, bias):
    logits = F.linear(x.float(), weight.float())
    if case.scoring == "sqrtsoftplus":
        scores = F.softplus(logits).sqrt()
    elif case.scoring == "sigmoid":
        scores = logits.sigmoid()
    else:
        scores = logits.softmax(-1)
    ranking = scores if bias is None else scores + bias
    if case.top_groups < case.groups:
        grouped = ranking.reshape(case.tokens, case.groups, -1)
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        groups = group_scores.topk(case.top_groups, dim=-1).indices
        mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
            1, groups, True
        )
        ranking = grouped.masked_fill(~mask.unsqueeze(-1), -torch.inf).flatten(1)
    return scores, ranking


def gate_weights(case, scores, ids):
    selected = scores.gather(-1, ids)
    return selected / (selected.sum(-1, keepdim=True) + 1e-20) * case.scale


def gate_reference(case, x, weight, bias):
    scores, ranking = gate_scores(case, x, weight, bias)
    ids = ranking.topk(case.topk, dim=-1).indices
    return ids, gate_weights(case, scores, ids)


def check_gate(case, inputs, outputs):
    ids, weights = (tensor.cpu() for tensor in outputs)
    assert ids.shape == weights.shape == (case.tokens, case.topk)
    assert ids.dtype in (torch.int32, torch.int64)
    assert weights.dtype == torch.float32
    ids = ids.long()
    assert ((ids >= 0) & (ids < case.experts)).all()
    ordered = ids.sort(-1).values
    assert (ordered[:, 1:] != ordered[:, :-1]).all()

    scores, ranking = gate_scores(case, *inputs)
    cutoff = ranking.topk(case.topk, dim=-1).values[:, -1:]
    # Ties can pick different IDs, but all returned experts must make the cut.
    assert (ranking.gather(-1, ids) >= cutoff - 2e-5).all()
    torch.testing.assert_close(
        weights, gate_weights(case, scores, ids), rtol=2e-4, atol=2e-5
    )


def prepare_gate_baseline_inputs(inputs):
    """Prepare the fixed FP32 weight from the shared BF16 input."""
    x, weight, bias = inputs
    return x, weight.float(), bias


def gate_baseline(case, x, weight_fp32, bias):
    import torch_npu

    logits = F.linear(x.float(), weight_fp32)
    kwargs = dict(
        k=case.topk,
        k_group=case.top_groups,
        group_count=case.groups,
        group_select_mode=1,
        renorm=int(case.scoring == "softmax"),
        norm_type={"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}[case.scoring],
        out_flag=False,
        routed_scaling_factor=case.scale,
        eps=1e-20,
    )
    if case.scoring == "sqrtsoftplus":
        weights, ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
            logits, bias=bias, **kwargs
        )
    else:
        weights, ids, _ = torch_npu.npu_moe_gating_top_k(logits, bias=bias, **kwargs)
    return ids, weights


def graph_operations(name, case, native, reference, candidate_kwargs, *, prepare=None):
    baseline = os.environ.get("FLAGGEMS_MEGA_BASELINE", "native")
    require_candidate = os.environ.get("FLAGGEMS_MEGA_REQUIRE_CANDIDATE") == "1"
    candidate = getattr(flaggems_vllm, name, None)
    assert baseline in ("native", "torch"), f"Unknown Mega baseline: {baseline}"
    assert callable(candidate) or (
        candidate is None and not require_candidate
    ), f"Missing public candidate: flaggems_vllm.{name}"

    if baseline == "native":
        from vllm_ascend.utils import enable_custom_op

        enabled = enable_custom_op()
        assert enabled, "vLLM-Ascend custom operators could not be loaded"
        yield "native", partial(native, case), prepare
    else:
        yield "torch_npu_composite", partial(reference, case), None

    if candidate is not None:
        yield "candidate", partial(candidate, **candidate_kwargs), None


def gate_kwargs(case):
    return dict(
        topk=case.topk,
        scoring=case.scoring,
        groups=case.groups,
        top_groups=case.top_groups,
        scale=case.scale,
    )


def gate_operations(case):
    return graph_operations(
        "mega_gate",
        case,
        gate_baseline,
        gate_reference,
        gate_kwargs(case),
        prepare=prepare_gate_baseline_inputs,
    )


def capture_graph(fn, cpu_inputs, *, stream=None):
    inputs = tuple(
        tensor.npu() if tensor is not None else None for tensor in cpu_inputs
    )
    torch.npu.synchronize()
    with nullcontext() if stream is None else torch.npu.stream(stream):
        fn(*inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        outputs = fn(*inputs)
    return graph, inputs, outputs


def flagtree_graph_smoke():
    import triton
    import triton.language as tl

    @triton.jit
    def add_one(x, y, size: tl.constexpr):
        offsets = tl.arange(0, size)
        tl.store(y + offsets, tl.load(x + offsets) + 1)

    def run(x, output):
        add_one[(1,)](x, output, 128)
        return output

    x = torch.arange(128, dtype=torch.float32)
    graph, (static_x, _), y = capture_graph(run, (x, torch.empty_like(x)))
    static_x.fill_(4)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), torch.full((128,), 5.0))
    graph.reset()


@torch.inference_mode()
def validate_graph(fn, case, make_inputs, check, *, prepare=None, stream=None):
    original = make_inputs(case, 2026)
    static_inputs = prepare(original) if prepare is not None else original
    graph, inputs, outputs = capture_graph(fn, static_inputs, stream=stream)
    for seed in (2026, 2027, 2028, 2026):
        if seed == 2026:
            cpu_inputs, prepared = original, static_inputs
        else:
            cpu_inputs = make_inputs(case, seed)
            prepared = prepare(cpu_inputs) if prepare is not None else cpu_inputs
        for dst, src in zip(inputs, prepared):
            if dst is not None:
                dst.copy_(src)
        graph.replay()
        torch.npu.synchronize()
        check(case, cpu_inputs, outputs)
    graph.reset()


@torch.inference_mode()
def time_graphs(operations, cpu_inputs, warmup, iterations, rounds=3, *, stream=None):
    assert warmup >= 0 and iterations > 0 and rounds > 0, "Invalid Graph timing counts"
    # Keep captured inputs and outputs alive for all replays.
    graphs = []
    for name, fn, prepare in operations:
        inputs = prepare(cpu_inputs) if prepare is not None else cpu_inputs
        graphs.append((name, *capture_graph(fn, inputs, stream=stream)))
    measurements = {name: [] for name, *_ in graphs}
    for _, graph, _, _ in graphs:
        for _ in range(warmup):
            graph.replay()
    torch.npu.synchronize()
    for index in range(rounds):
        order = graphs if index % 2 == 0 else reversed(graphs)
        for name, graph, _, _ in order:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            host_start = perf_counter_ns()
            start.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            torch.npu.synchronize()
            measurements[name].append(
                {
                    "event_us": start.elapsed_time(end) * 1000 / iterations,
                    "host_us": (perf_counter_ns() - host_start) / 1000 / iterations,
                }
            )
    for _, graph, _, _ in graphs:
        graph.reset()
    return measurements


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = load_gate_cases(quick=metafunc.config.getoption("--quick"))
        metafunc.parametrize("case", cases, ids=[case.id for case in cases])


def test_mega_gate_graph(case):
    for _, fn, prepare in gate_operations(case):
        validate_graph(fn, case, gate_inputs, check_gate, prepare=prepare)


def test_mega_gate_prepared_streams():
    case = GateCase("prepared", 5, 2048, 256, 8, "softmax")
    kwargs = gate_kwargs(case)
    first_cpu = gate_inputs(case, 2026)
    second_cpu = gate_inputs(case, 2027)
    first = tuple(t.npu() if t is not None else None for t in first_cpu)
    second = tuple(t.npu() if t is not None else None for t in second_cpu)
    default = torch.npu.current_stream()
    other = torch.npu.Stream()
    prepared = flaggems_vllm.prepare_mega_gate(*first, **kwargs)
    other.wait_stream(default)
    with torch.npu.stream(other):
        with pytest.raises(AssertionError, match="current stream"):
            flaggems_vllm.mega_gate(*second, prepared=prepared, **kwargs)
        other_prepared = flaggems_vllm.prepare_mega_gate(*second, **kwargs)
        other_outputs = flaggems_vllm.mega_gate(
            *second, prepared=other_prepared, **kwargs
        )
    first_outputs = flaggems_vllm.mega_gate(*first, prepared=prepared, **kwargs)
    second_outputs = flaggems_vllm.mega_gate(*second, prepared=prepared, **kwargs)
    torch.npu.synchronize()
    check_gate(case, first_cpu, first_outputs)
    check_gate(case, second_cpu, second_outputs)
    check_gate(case, second_cpu, other_outputs)
    assert all(
        a.data_ptr() != b.data_ptr() for a, b in zip(first_outputs, second_outputs)
    )
    assert first_outputs[0].dtype == second_outputs[0].dtype == torch.int64
    with pytest.raises(AssertionError, match="prepared shape"):
        flaggems_vllm.mega_gate(first[0][:-1], *first[1:], prepared=prepared, **kwargs)
    with pytest.raises(AssertionError, match="routing parameters"):
        flaggems_vllm.mega_gate(*first, prepared=prepared, **dict(kwargs, scale=2.0))
    validate_graph(
        partial(flaggems_vllm.mega_gate, prepared=other_prepared, **kwargs),
        case,
        gate_inputs,
        check_gate,
        stream=other,
    )


def test_mega_gate_v4_flash_no_bias_graph():
    case = GateCase("v4-flash", 5, 4096, 256, 6, "sqrtsoftplus", scale=1.5)

    def inputs(case, seed):
        x, weight, _ = gate_inputs(case, seed)
        if seed == 2026:
            x.zero_()
        return x, weight, None

    def check(case, inputs, outputs):
        check_gate(case, inputs, outputs)
        if not inputs[0].count_nonzero():
            expected = torch.arange(case.topk).expand(case.tokens, -1)
            torch.testing.assert_close(outputs[0].cpu(), expected)

    validate_graph(
        partial(flaggems_vllm.mega_gate, **gate_kwargs(case)), case, inputs, check
    )


@pytest.mark.parametrize(
    "geometry",
    [
        GateCase("all-groups", 7, 384, 48, 5, "sigmoid", groups=48, top_groups=48),
        GateCase("small-weights", 2048, 4096, 128, 5, "softmax"),
    ],
    ids=["all-groups", "small-weights"],
)
def test_mega_gate_general_shapes_graph(geometry):
    validate_graph(
        partial(flaggems_vllm.mega_gate, **gate_kwargs(geometry)),
        geometry,
        gate_inputs,
        check_gate,
    )


@pytest.fixture
def empty_gate_inputs():
    x = torch.empty((0, 4096), dtype=torch.bfloat16, device=flaggems_vllm.device)
    weight = torch.empty((256, 4096), dtype=torch.bfloat16, device=x.device)
    return x, weight


def test_mega_gate_empty(empty_gate_inputs):
    ids, weights = flaggems_vllm.mega_gate(*empty_gate_inputs)
    assert ids.shape == weights.shape == (0, 6)
    assert ids.dtype == torch.int64 and weights.dtype == torch.float32


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"topk": 6.0}, "must be integers"),
        ({"topk": True}, "must be integers"),
        ({"topk": 0}, "topk must be between"),
        ({"scoring": "invalid"}, "Unsupported MegaGate scoring"),
        ({"groups": 0}, "groups must divide experts"),
        ({"groups": 64, "top_groups": 1}, "at least topk experts"),
        ({"scale": float("inf")}, "scale must be finite"),
        ({"prepared": object()}, "Invalid prepared MegaGate handle"),
    ],
)
def test_mega_gate_invalid_parameters(empty_gate_inputs, kwargs, message):
    with pytest.raises(AssertionError, match=message):
        flaggems_vllm.mega_gate(*empty_gate_inputs, **kwargs)
