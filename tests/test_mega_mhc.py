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

import importlib
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from unittest.mock import Mock

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
    # Keep BF16 activation rounding separate from absolute FP32 coefficient bounds.
    # The native pipeline and fused kernel are both checked against the CPU reference.
    names = ("current", "post", "comb", "output", "pre")
    tolerances = ((1e-2, 2e-2), (0, 6e-5), (0, 3e-5), (1e-2, 2e-2), (0, 2e-5))
    assert len(actual) == len(expected)
    for name, value, ref, (rtol, atol) in zip(names, actual, expected, tolerances):
        torch.testing.assert_close(
            value, ref, rtol=rtol, atol=atol, msg=lambda msg: f"{name}: {msg}"
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


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = load_mhc_cases(quick=metafunc.config.getoption("--quick"))
        metafunc.parametrize("case", cases, ids=[case.id for case in cases])


def test_mega_mhc_graph(case):
    for _, fn, prepare in mhc_operations(case):
        validate_graph(fn, case, mhc_inputs, check_mhc, prepare=prepare)


@torch.inference_mode()
def _stress_pair_local_replay():
    from triton.runtime import driver

    from flaggems_vllm.runtime.backend._ascend.ops.mega_mhc import _launch_config

    utils = driver.active.utils
    cores = utils.get_aicore_num()
    graphs = []
    try:
        # Both originally used inter-program split-K. The shifted case also
        # leaves a partial tile and unequal work across the participating pairs.
        for case in (MhcCase("normal", 17, 4096), MhcCase("shifted", 73, 5120, True)):
            (workers,), _, opts = _launch_config(
                case.tokens,
                case.hidden,
                case.shifted,
                cores,
                utils.get_aivector_core_num(),
            )
            assert 1 < workers <= cores
            if case.shifted:
                assert case.tokens % opts["BLOCK_M"] != 0
            stream = torch.npu.Stream()
            fn = partial(flaggems_vllm.mega_mhc, rms_eps=mhc_eps(case))
            graph, inputs, outputs = capture_graph(
                fn, mhc_inputs(case, 3600), stream=stream
            )
            graphs.append((case, stream, graph, inputs, outputs))
        for seed in (*range(3600, 3616), 3600):
            changed = [mhc_inputs(case, seed) for case, *_ in graphs]
            for (_, stream, _, inputs, _), values in zip(graphs, changed):
                with torch.npu.stream(stream):
                    for dst, src in zip(inputs, values):
                        if dst is not None:
                            dst.copy_(src)
            # Enqueue on both streams before waiting on either one.
            for _, stream, graph, _, _ in graphs:
                with torch.npu.stream(stream):
                    graph.replay()
            torch.npu.synchronize()
            for (case, _, _, _, outputs), values in zip(graphs, changed):
                check_mhc(case, values, outputs)
    finally:
        for _, _, graph, _, _ in graphs:
            graph.reset()


def test_mega_mhc_cross_program_replay():
    # A device-side barrier can block synchronize() indefinitely. Bound the child
    # from a separate process so the deadline also applies inside native code.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_mega_mhc import _stress_pair_local_replay;"
            " _stress_pair_local_replay()",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=300,
    )


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


@pytest.mark.parametrize("shifted,tokens", [(False, 16), (True, 16), (False, 496)])
@torch.inference_mode()
def test_mega_mhc_prepared_context(shifted, tokens):
    case = MhcCase("prepared", tokens, 5120 if shifted else 4096, shifted=shifted)
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
            # An ordinary context must read an in-place weight update on replay.
            changed = (*changed[:4], new[4], *changed[5:])
            for dst, before, after in zip(static_inputs, original, changed):
                if after is not before:
                    dst.copy_(after)
            graph.replay()
            torch.npu.synchronize()
            check_mhc(case, changed, result)
        finally:
            graph.reset()


@torch.inference_mode()
def test_mega_mhc_shared_packed_weights(monkeypatch):
    assert not hasattr(flaggems_vllm, "pack_mega_mhc_weights")
    cases = [MhcCase("packed", 496, 4096), MhcCase("packed", 513, 4096, True)]
    original = [mhc_inputs(case, 2000) for case in cases]
    fn_cpu = original[0][4]
    fn = fn_cpu.npu()
    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.ops.mega_mhc"
    )
    pack = Mock(wraps=module._write_projection)
    monkeypatch.setattr(module, "_write_projection", pack)
    short_case = MhcCase("direct", 1, 4096)
    short = _to_npu(mhc_inputs(short_case, 2000))
    direct = flaggems_vllm.create_mega_mhc_context(
        *short[:4], fn, *short[5:], rms_eps=mhc_eps(short_case), fixed_weights=True
    )
    direct.refresh_weights()
    pack.assert_not_called()
    fixed_context = None
    planes = None
    torch.npu.current_stream().synchronize()
    graphs = []
    try:
        # A dynamic context created after the two fixed contexts must still
        # observe fn.copy_() immediately, including in a previously captured graph.
        for index, fixed in ((0, True), (1, True), (0, False)):
            case, cpu = cases[index], original[index]
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                inputs = (*_to_npu(cpu[:4]), fn, *_to_npu(cpu[5:]))
                context = flaggems_vllm.create_mega_mhc_context(
                    *inputs, rms_eps=mhc_eps(case), fixed_weights=fixed
                )
                buffer = context._launch_tail[3]
                if planes is None:
                    fixed_context = context
                    planes = buffer
                    high = fn_cpu.bfloat16()
                    expected = torch.zeros((64, fn_cpu.shape[1]), dtype=torch.bfloat16)
                    expected[:24] = high
                    expected[32:56] = (fn_cpu - high.float()).bfloat16()
                    assert torch.equal(planes.cpu(), expected)
                assert (buffer is planes) == fixed
                call = partial(
                    flaggems_vllm.mega_mhc,
                    *inputs,
                    rms_eps=mhc_eps(case),
                    context=context,
                )
                call()
                stream.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    outputs = call()
                graphs.append((graph, stream, call, outputs, index, fixed))
        assert pack.call_count == 1
        with pytest.raises(AssertionError, match="requires a fixed-weight context"):
            context.refresh_weights()  # The last context is dynamic.
        updated = fn_cpu.neg()
        fn.copy_(updated)
        for refreshed in (False, True):
            if refreshed:
                fixed_context.refresh_weights()
                assert pack.call_count == 2
            for graph, stream, *_ in graphs:
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    graph.replay()
            torch.npu.synchronize()
            for _, _, _, outputs, index, fixed in graphs:
                cpu = original[index]
                weight = fn_cpu if fixed and not refreshed else updated
                check_mhc(cases[index], (*cpu[:4], weight, *cpu[5:]), outputs)
    finally:
        for graph, *_ in graphs:
            graph.reset()
