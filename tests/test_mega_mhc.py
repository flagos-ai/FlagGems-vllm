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
    pytest.mark.mega_mhc,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "ascend", reason="requires Ascend ACL Graph"
    ),
]


@dataclass(frozen=True)
class MhcCase:
    profile: str
    tokens: int
    hidden: int
    shifted: bool = False

    @property
    def id(self):
        return f"{self.profile}-t{self.tokens}"


def load_mhc_cases(*, quick=False, shape_file=None):
    path = (
        Path(shape_file)
        if shape_file
        else (Path(__file__).resolve().parents[1] / "benchmark" / "core_shapes.yaml")
    )
    config = yaml.safe_load(path.read_text())["mega_mhc"]
    cases = [MhcCase(*shape) for shape in config["shapes"]]
    if quick and "quick_tokens" in config:
        tokens = set(config["quick_tokens"])
        cases = [case for case in cases if case.tokens in tokens]
    assert cases, f"no MegaMHC shapes selected from {path}"
    return cases


def mhc_inputs(case, seed):
    rng = torch.Generator().manual_seed(seed)
    t, h = case.tokens, case.hidden
    x = torch.randn(t, h, dtype=torch.bfloat16, generator=rng)
    residual = torch.randn(t, 4, h, dtype=torch.bfloat16, generator=rng)
    post = torch.rand(t, 4, generator=rng)
    comb = torch.rand(t, 4, 4, generator=rng).softmax(-1)
    fn = torch.randn(24, 4 * h, generator=rng).mul_((4 * h) ** -0.5)
    scale = torch.tensor([0.1, 0.1, 0.1])
    base = torch.randn(24, generator=rng).mul_(0.1)
    norm_weight = torch.randn(h, generator=rng).mul_(0.1).add_(1).bfloat16()
    pre_mix = torch.rand(t, 4, generator=rng) if case.shifted else None
    return x, residual, post, comb, fn, scale, base, norm_weight, pre_mix


def mhc_eps(case):
    return 1e-20 if case.shifted else 1e-6


def _to_npu(inputs):
    return tuple(value.npu() if value is not None else None for value in inputs)


def mhc_reference(case, x, residual, post, comb, fn, scale, base, norm_weight, pre_mix):
    eps = 1e-6
    rms_eps = mhc_eps(case)
    post_shape = post.shape
    current = torch.bmm(comb.transpose(1, 2), residual.float())
    current = (current + post.reshape(-1, 4, 1) * x.float().unsqueeze(1)).bfloat16()
    flat = current.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, True) + rms_eps)
    pre, post, comb = mixes.split((4, 4, 16), dim=-1)
    pre = (pre * scale[0] + base[:4]).sigmoid() + eps
    post = (post * scale[1] + base[4:8]).sigmoid() * 2
    comb = (comb * scale[2] + base[8:]).view(-1, 4, 4).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    applied_pre = pre if pre_mix is None else pre_mix
    layer = (applied_pre.unsqueeze(-1) * current.float()).sum(1).bfloat16().float()
    layer = layer * torch.rsqrt(layer.square().mean(-1, True) + rms_eps)
    result = (
        current,
        post.reshape(post_shape),
        comb,
        (layer * norm_weight.float()).bfloat16(),
    )
    return result if pre_mix is None else (*result, pre)


def check_mhc(case, inputs, outputs, *, exact_comb=False):
    actual = tuple(tensor.cpu() for tensor in outputs)
    expected = mhc_reference(case, *inputs)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-2,
        atol=2e-2,
    )
    if exact_comb:
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=1e-5)


def mhc_baseline(case, x, residual, post, comb, fn, scale, base, norm_weight, pre_mix):
    import torch_npu

    ops = torch.ops._C_ascend
    current = ops.npu_hc_post(
        x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
    ).squeeze(0)
    layer, new_post, new_comb, new_pre = ops.npu_hc_pre_v3(
        current,
        fn,
        scale,
        base,
        pre_mix,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        norm_eps=mhc_eps(case),
        hc_eps=1e-6,
    )
    layer, _ = torch_npu.npu_rms_norm(layer, norm_weight, epsilon=mhc_eps(case))
    result = (current, new_post, new_comb, layer)
    return result if pre_mix is None else (*result, new_pre)


def mhc_operations(case):
    baseline = os.environ.get("FLAGGEMS_MEGA_BASELINE", "native")
    require_candidate = os.environ.get("FLAGGEMS_MEGA_REQUIRE_CANDIDATE") == "1"
    candidate = getattr(flaggems_vllm, "mega_mhc", None)
    assert baseline in ("native", "torch"), f"Unknown Mega baseline: {baseline}"
    assert callable(candidate) or (
        candidate is None and not require_candidate
    ), "Missing public candidate: flaggems_vllm.mega_mhc"

    if baseline == "native":
        from vllm_ascend.utils import enable_custom_op

        enabled = enable_custom_op()
        assert enabled, "vLLM-Ascend custom operators could not be loaded"
        yield "native", partial(mhc_baseline, case), None
    else:
        yield "torch_npu_composite", partial(mhc_reference, case), None

    if candidate is not None:
        yield "candidate", partial(candidate, rms_eps=mhc_eps(case)), None


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
        cases = load_mhc_cases(quick=metafunc.config.getoption("--quick"))
        metafunc.parametrize("case", cases, ids=[case.id for case in cases])


def test_mega_mhc_graph(case):
    for _, fn, prepare in mhc_operations(case):
        validate_graph(fn, case, mhc_inputs, check_mhc, prepare=prepare)


@pytest.mark.parametrize("tokens,hidden", [(384, 3968), (704, 2048), (4096, 2048)])
def test_mega_mhc_vector_tile_geometry(tokens, hidden):
    case = MhcCase("vector-tile", tokens, hidden)
    validate_graph(flaggems_vllm.mega_mhc, case, mhc_inputs, check_mhc)


@pytest.mark.parametrize("tokens", [192, 512])
def test_mega_mhc_slow_sinkhorn_graph(tokens):
    case = MhcCase("slow-sinkhorn", tokens, 4096)

    def inputs(case, seed):
        values = mhc_inputs(case, seed)
        if seed != 2026:
            # Keep the comb logits independent of projection rounding.
            values[5][2] = 0.0
            values[6][8:] = torch.tensor(
                [12, 2, -8, -4, 10, 0, 2, -8, -4, 5, 10, 0, 2, -2, 5, 0]
            )
        return values

    def check(case, values, outputs):
        check_mhc(case, values, outputs, exact_comb=bool(values[5][2] == 0))

    def run(*values):
        return flaggems_vllm.mega_mhc(*values, rms_eps=mhc_eps(case))

    validate_graph(run, case, inputs, check)


@pytest.mark.parametrize("shifted", [False, True])
def test_mega_mhc_empty_and_invalid_shape(shifted):
    case = MhcCase("empty", 0, 4096, shifted=shifted)
    cpu_inputs = mhc_inputs(case, 2026)
    inputs = _to_npu(cpu_inputs)
    result = flaggems_vllm.mega_mhc(*inputs)
    check_mhc(case, cpu_inputs, result)
    with pytest.raises(AssertionError, match="residual must have shape"):
        flaggems_vllm.mega_mhc(inputs[0], inputs[1][:, :3], *inputs[2:])


@pytest.mark.parametrize("shifted", [False, True])
@torch.inference_mode()
def test_mega_mhc_prepared_context(shifted):
    case = MhcCase("prepared", 16, 5120 if shifted else 4096, shifted=shifted)
    eps = mhc_eps(case)
    original = mhc_inputs(case, 2026)
    if not shifted:
        original = (*original[:2], original[2].unsqueeze(-1), *original[3:])
    inputs = _to_npu(original)
    new = mhc_inputs(case, 2027)
    post = new[2].reshape_as(original[2])
    changed = (*new[:2], post, new[3], *original[4:8], new[8])
    changed_npu = (*_to_npu(changed[:4]), *inputs[4:8], *_to_npu(changed[8:]))
    context = flaggems_vllm.create_mega_mhc_context(*inputs, rms_eps=eps)
    first = flaggems_vllm.mega_mhc(*inputs, rms_eps=eps, context=context)
    second = flaggems_vllm.mega_mhc(*changed_npu, rms_eps=eps, context=context)
    torch.npu.synchronize()
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(first, second))
    check_mhc(case, original, first)
    check_mhc(case, changed, second)

    with pytest.raises(AssertionError, match="must match the prepared context"):
        flaggems_vllm.mega_mhc(*inputs, rms_eps=eps * 2, context=context)
    with pytest.raises(AssertionError, match="weights or their storage changed"):
        flaggems_vllm.mega_mhc(
            *inputs[:4], inputs[4].clone(), *inputs[5:], rms_eps=eps, context=context
        )
    with torch.npu.stream(torch.npu.Stream()):
        with pytest.raises(AssertionError, match="context belongs to another stream"):
            flaggems_vllm.mega_mhc(*inputs, rms_eps=eps, context=context)

    # A contiguous slice can have an address that does not match the prepared kernel.
    unaligned = torch.empty(
        inputs[0].numel() + 1, device=inputs[0].device, dtype=inputs[0].dtype
    )
    unaligned = unaligned[1:].view_as(inputs[0])
    with pytest.raises(AssertionError, match="tensor alignment"):
        flaggems_vllm.mega_mhc(unaligned, *inputs[1:], rms_eps=eps, context=context)

    stream = torch.npu.Stream()
    graph_context = None

    def run(*values):
        nonlocal graph_context
        if graph_context is None:
            graph_context = flaggems_vllm.create_mega_mhc_context(*values, rms_eps=eps)
        return flaggems_vllm.mega_mhc(*values, rms_eps=eps, context=graph_context)

    graph, static_inputs, result = capture_graph(run, original, stream=stream)
    with torch.npu.stream(stream):
        try:
            graph.replay()
            torch.npu.synchronize()
            check_mhc(case, original, result)
            for dst, before, after in zip(static_inputs, original, changed):
                if after is not before:
                    dst.copy_(after)
            graph.replay()
            torch.npu.synchronize()
            check_mhc(case, changed, result)
        finally:
            graph.reset()
