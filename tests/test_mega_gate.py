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
from itertools import combinations
from pathlib import Path

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
    def input_dtype(self):
        if self.profile in ("v32", "kimi-k25", "qwen-35b", "qwen-397b"):
            return torch.bfloat16
        return torch.float32

    @property
    def logits_dtype(self):
        if self.profile in ("qwen-35b", "qwen-397b"):
            return torch.bfloat16
        return torch.float32

    @property
    def id(self):
        return f"{self.profile}-t{self.tokens}"


GATE_SHAPE_FILE = Path(__file__).resolve().parents[1] / "benchmark/core_shapes.yaml"


def load_gate_cases(shape_file=GATE_SHAPE_FILE, *, quick=False):
    with open(shape_file) as stream:
        config = (yaml.safe_load(stream) or {}).get("mega_gate")
    assert config, f"No mega_gate configuration in {shape_file}"
    cases = [GateCase(*shape) for shape in config["shapes"]]
    if quick:
        first_by_model = {}
        for case in cases:
            first_by_model.setdefault(case.profile, case)
        cases = list(first_by_model.values())
    assert cases, f"No matching mega_gate shapes in {shape_file}"
    return cases


def gate_inputs(case, seed):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(case.tokens, case.hidden, generator=rng, dtype=case.input_dtype)
    weight = (
        torch.randn(case.experts, case.hidden, generator=rng)
        .mul_(case.hidden**-0.5)
        .to(case.input_dtype)
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


def _valid_group_tie(case, ranking, ids):
    # The FP32 matmuls can disagree on a near-tie between the last admitted
    # group and the first excluded group. Require one legal set of whole
    # groups, then check every returned expert against that set's cutoff.
    group_tol = 2e-6
    width = case.experts // case.groups
    group_scores = ranking.reshape(case.groups, width).topk(2).values.sum(-1)
    group_cutoff = group_scores.topk(case.top_groups).values[-1]
    mandatory = set(torch.where(group_scores > group_cutoff + group_tol)[0].tolist())
    mandatory.update((ids // width).tolist())
    allowed = set(torch.where(group_scores >= group_cutoff - group_tol)[0].tolist())
    if len(mandatory) > case.top_groups or not mandatory <= allowed:
        return False
    for extra in combinations(
        sorted(allowed - mandatory), case.top_groups - len(mandatory)
    ):
        chosen = sorted(mandatory.union(extra))
        outside = sorted(set(range(case.groups)) - set(chosen))
        if group_scores[chosen].min() < group_scores[outside].max() - group_tol:
            continue
        omitted = ranking.clone()
        omitted[ids] = -torch.inf
        best_omitted = omitted.reshape(case.groups, width)[chosen].max()
        if ranking[ids].min() >= best_omitted - 2e-5:
            return True
    return False


def _check_qwen_bf16(case, inputs, ids, weights):
    x, weight, bias = inputs
    assert case.scoring == "softmax" and bias is None and case.groups == 1
    raw = F.linear(x.float(), weight.float())
    # FP32 reduction orders can land on opposite sides of a BF16 midpoint.
    # For these unit-variance logits, only bins reachable within 1e-5 of
    # the independent dot product are admissible.
    low = (raw - 1e-5).bfloat16().float()
    high = (raw + 1e-5).bfloat16().float()
    best_omitted = low.scatter(1, ids, -torch.inf).amax(-1, keepdim=True)
    low, high = low.gather(1, ids), high.gather(1, ids)
    assert (high >= best_omitted).all(), "expert outside BF16 cutoff"
    # Bound each selected probability with its own low (high) logit and
    # all other selected logits at their high (low) values.
    shift = high.amax(-1, keepdim=True)
    lo, hi = (low - shift).exp(), (high - shift).exp()
    lower = case.scale * lo / (hi.sum(-1, keepdim=True) - hi + lo)
    upper = case.scale * hi / (lo.sum(-1, keepdim=True) - lo + hi)
    assert (weights >= (lower - 2e-6).to(weights.dtype)).all()
    assert (weights <= (upper + 2e-6).to(weights.dtype)).all()


def check_gate(case, inputs, outputs, *, native=False):
    ids, weights = (tensor.cpu() for tensor in outputs)
    assert ids.shape == weights.shape == (case.tokens, case.topk)
    expected_ids = (
        torch.int32 if native or inputs[1].dtype == torch.float32 else torch.int64
    )
    assert ids.dtype == expected_ids
    assert weights.dtype == (case.logits_dtype if native else torch.float32)
    ids = ids.long()
    assert ((ids >= 0) & (ids < case.experts)).all()
    ordered = ids.sort(-1).values
    assert (ordered[:, 1:] != ordered[:, :-1]).all()

    if case.logits_dtype == torch.bfloat16:
        _check_qwen_bf16(case, inputs, ids, weights)
        return
    scores, ranking = gate_scores(case, *inputs)
    cutoff = ranking.topk(case.topk, dim=-1).values[:, -1:]
    # Ties can pick different IDs, but all dynamic experts must make the cut.
    valid = (ranking.gather(-1, ids) >= cutoff - 2e-5).all(-1)
    bad_rows = torch.where(~valid)[0].tolist()
    for row in bad_rows:
        assert case.top_groups < case.groups, f"expert cutoff mismatch at row {row}"
        row_scores = scores[row] if inputs[2] is None else scores[row] + inputs[2]
        assert _valid_group_tie(
            case, row_scores, ids[row]
        ), f"group or expert cutoff mismatch at row {row}"

    torch.testing.assert_close(
        weights, gate_weights(case, scores, ids).to(weights.dtype), rtol=2e-4, atol=2e-5
    )


def gate_baseline(
    case,
    x,
    weight,
    bias,
    *,
    input_ids=None,
    tid2eid=None,
    bias_vl=None,
    image_sentinel_lo=129257,
):
    if case.profile in ("v32", "kimi-k25"):
        # The BF16 fast path in AscendGateLinear.forward.
        logits = torch.mm(x, weight.t(), out_dtype=torch.float32)
    elif case.scoring == "sqrtsoftplus":
        logits = F.linear(x.float(), weight)
    else:
        # Qwen's unquantized ReplicatedLinear retains BF16 output rounding.
        logits = F.linear(x, weight)
    kwargs = dict(
        k=case.topk,
        k_group=case.top_groups,
        group_count=case.groups,
        group_select_mode=1,
        renorm=int(case.scoring != "sqrtsoftplus"),
        norm_type={"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}[case.scoring],
        out_flag=False,
        routed_scaling_factor=case.scale,
        eps=1e-20,
    )
    if case.scoring == "sqrtsoftplus":
        if input_ids is not None:
            input_ids = torch.where(input_ids == -1, 0, input_ids.long())
        if bias_vl is not None:
            from vllm_ascend.device.hardware_profile import (
                HardwareCapability,
                get_current_hardware_profile,
            )

            fused_vision = get_current_hardware_profile().supports(
                HardwareCapability.MOE_GATING_TOP_K_HASH_VISION
            )
        if bias_vl is not None and not fused_vision:
            from vllm_ascend.ops.fused_moe.router.fused_topk_router import (
                select_deepseek_v4_vision_experts,
            )

            weights, ids = select_deepseek_v4_vision_experts(
                logits,
                input_ids,
                tid2eid,
                bias_vl,
                bias,
                case.topk,
                True,
                case.scale,
                image_sentinel_lo,
            )
            return ids.int(), weights
        if bias_vl is not None:
            kwargs.update(
                bias_vl=bias_vl,
                image_sentinel_lo=image_sentinel_lo,
                image_sentinel_count=5,
            )
        weights, ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
            logits, bias=bias, input_ids=input_ids, tid2eid=tid2eid, **kwargs
        )
    else:
        # Initialize the native ops package before its device adaptor.
        import vllm_ascend.ops  # noqa: F401
        from vllm_ascend.device.device_op import DeviceOperator

        weights, ids, _ = DeviceOperator.moe_gating_top_k(
            logits, bias_opt=bias, **kwargs
        )
    return ids.int(), weights


def gate_kwargs(case):
    return dict(
        topk=case.topk,
        scoring=case.scoring,
        logits_dtype=case.logits_dtype,
        groups=case.groups,
        top_groups=case.top_groups,
        scale=case.scale,
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


@torch.inference_mode()
def validate_graph(fn, case, make_inputs, check, *, stream=None):
    original = make_inputs(case, 2026)
    graph, inputs, outputs = capture_graph(fn, original, stream=stream)
    for seed in (2026, 2027, 2028, 2026):
        cpu_inputs = original if seed == 2026 else make_inputs(case, seed)
        for dst, src in zip(inputs, cpu_inputs):
            if dst is not None:
                dst.copy_(src)
        graph.replay()
        torch.npu.synchronize()
        check(case, cpu_inputs, outputs)
    graph.reset()


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = load_gate_cases(
            os.environ.get("MEGA_GATE_SHAPE_FILE", GATE_SHAPE_FILE),
            quick=metafunc.config.getoption("--quick"),
        )
        metafunc.parametrize("case", cases, ids=[case.id for case in cases])


def test_mega_gate_graph(case):
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
    validate_graph(
        partial(gate_baseline, case),
        case,
        gate_inputs,
        partial(check_gate, native=True),
    )
    validate_graph(
        partial(flaggems_vllm.mega_gate, **gate_kwargs(case)),
        case,
        gate_inputs,
        check_gate,
    )


def test_mega_gate_prepared_streams():
    from flaggems_vllm.runtime.backend._ascend.ops.mega_gate import prepare_mega_gate

    case = GateCase("prepared", 5, 2048, 256, 8, "softmax")
    kwargs = gate_kwargs(case)
    first_cpu = gate_inputs(case, 2026)
    second_cpu = gate_inputs(case, 2027)
    first = tuple(t.npu() if t is not None else None for t in first_cpu)
    second = tuple(t.npu() if t is not None else None for t in second_cpu)
    default = torch.npu.current_stream()
    other = torch.npu.Stream()
    prepared = prepare_mega_gate(*first, **kwargs)
    other.wait_stream(default)
    with torch.npu.stream(other):
        with pytest.raises(AssertionError, match="current stream"):
            flaggems_vllm.mega_gate(*second, prepared=prepared, **kwargs)
        other_prepared = prepare_mega_gate(*second, **kwargs)
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
    assert first_outputs[0].dtype == second_outputs[0].dtype == torch.int32
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
            expected = torch.arange(case.topk, dtype=torch.int32).expand(
                case.tokens, -1
            )
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
        ({"output_dtypes": (torch.int8, torch.float32)}, "output_dtypes"),
        ({"logits_dtype": torch.float16}, "logits_dtype"),
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


def v4_route_inputs(case, mode, device):
    # Include padding, text, and the endpoints of the five image sentinels.
    ids = torch.tensor([129257, 1, 2, -1, 129256, 129261, 129262], device=device)
    ids = ids.repeat((case.tokens + 6) // 7)[: case.tokens]
    table = None
    if "hash" in mode:
        table = (
            torch.arange(129263, dtype=torch.int32, device=device)[:, None]
            + torch.arange(case.topk, dtype=torch.int32, device=device)
        ) % case.experts
    bias_vl = None
    if "vision" in mode:
        bias_vl = torch.linspace(-3, 3, case.experts, device=device)
    return dict(input_ids=ids, tid2eid=table, bias_vl=bias_vl)


@pytest.mark.parametrize("mode", ["hash", "vision", "vision-hash"])
@torch.inference_mode()
def test_mega_gate_v4_routes(mode):
    from vllm_ascend.utils import enable_custom_op

    from flaggems_vllm.runtime.backend._ascend.ops.mega_gate import prepare_mega_gate

    assert enable_custom_op()
    case = GateCase("v4-flash", 7, 4096, 256, 6, "sqrtsoftplus", scale=1.5)
    x, weight, bias = (t.npu() for t in gate_inputs(case, 2026))
    if "hash" in mode:
        bias = None
    routing = v4_route_inputs(case, mode, x.device)
    kwargs = dict(**gate_kwargs(case), **routing)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        handle = prepare_mega_gate(x, weight, bias, **kwargs)

    def call():
        return flaggems_vllm.mega_gate(x, weight, bias, prepared=handle, **kwargs)

    with torch.npu.stream(stream):
        call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        actual = call()
    for offset in (0, 1):
        routing["input_ids"][1] = 1 + offset
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        expected = gate_baseline(case, x, weight, bias, **routing)
        for outputs in (actual, flaggems_vllm.mega_gate(x, weight, bias, **kwargs)):
            torch.testing.assert_close(outputs[0], expected[0])
            torch.testing.assert_close(outputs[1], expected[1], rtol=2e-4, atol=2e-5)
            if routing["tid2eid"] is not None:
                assert outputs[0][1, 0].item() == 1 + offset
    graph.reset()


def test_mega_gate_v4_fp32_cutoff():
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
    case = GateCase("v4-flash", 2, 4096, 256, 6, "sqrtsoftplus", scale=1.5)
    x = torch.zeros((2, case.hidden), device="npu")
    weight = torch.zeros((case.experts, case.hidden), device="npu")
    # Experts 5 and 6 straddle K. Their activation difference vanishes in BF16.
    x[:, :2] = torch.tensor([[1.0, 1.001], [1.001, 1.0]], device=x.device)
    weight[:5, 0] = torch.arange(2, 7, device=x.device)
    weight[5, 0] = weight[6, 1] = 1
    actual = flaggems_vllm.mega_gate(x, weight, **gate_kwargs(case))
    expected = gate_baseline(case, x, weight, None)
    assert expected[0][:, -1].tolist() == [6, 5]
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_mega_gate_bf16_activation(weight_dtype):
    from flaggems_vllm.runtime.backend._ascend.ops.mega_gate import prepare_mega_gate

    case = GateCase("v4-flash", 5, 4096, 256, 6, "sqrtsoftplus", scale=1.5)
    x, weight, bias = gate_inputs(case, 2028)
    inputs = (x.bfloat16(), weight.to(weight_dtype), bias)
    tensors = tuple(t.npu() for t in inputs)
    handle = prepare_mega_gate(*tensors, **gate_kwargs(case))
    for prepared in (None, handle):
        result = flaggems_vllm.mega_gate(
            *tensors, prepared=prepared, **gate_kwargs(case)
        )
        check_gate(case, inputs, result)


def test_mega_gate_prepared_route_dtype():
    from flaggems_vllm.runtime.backend._ascend.ops.mega_gate import prepare_mega_gate

    case = GateCase("v4-flash", 3, 128, 16, 2, "sqrtsoftplus")
    x, weight, bias = (t.npu() for t in gate_inputs(case, 2026))
    routing = v4_route_inputs(case, "hash", x.device)
    kwargs = dict(**gate_kwargs(case), **routing)
    handle = prepare_mega_gate(x, weight, bias, **kwargs)
    kwargs["input_ids"] = routing["input_ids"].int()
    with pytest.raises(AssertionError, match="dtypes"):
        flaggems_vllm.mega_gate(x, weight, bias, prepared=handle, **kwargs)


def test_mega_gate_group_cutoff_tie():
    case = GateCase("group-tie", 1, 128, 6, 2, "sigmoid", 3, 1)
    ranking = torch.tensor([0.9, 0.8, 0.95, 0.750001, 0.6, 0.7])
    assert _valid_group_tie(case, ranking, torch.tensor([0, 1]))
    assert _valid_group_tie(case, ranking, torch.tensor([2, 3]))
    assert not _valid_group_tie(case, ranking, torch.tensor([0, 2]))
    assert not _valid_group_tie(case, ranking, torch.tensor([4, 5]))


def test_mega_gate_qwen_logits_rounding():
    from vllm_ascend.utils import enable_custom_op

    from flaggems_vllm.runtime.backend._ascend.ops.mega_gate import prepare_mega_gate

    assert enable_custom_op()
    case = GateCase("qwen-35b", 2, 128, 16, 2, "softmax")
    x = torch.zeros((2, 128), dtype=torch.bfloat16, device="npu")
    weight = torch.zeros((16, 128), dtype=x.dtype, device=x.device)
    # 1 and 1 + 2**-10 are distinct FP32 dot products but the same BF16 logit.
    x[:, 0] = 1
    x[:, 1] = torch.tensor([2**-10, -(2**-10)], dtype=x.dtype, device=x.device)
    weight[0, 0] = 2
    weight[1:3, 0] = 1
    weight[2, 1] = 1
    kwargs = gate_kwargs(case)
    native_ids, native_weights = gate_baseline(case, x, weight, None)
    handle = prepare_mega_gate(x, weight, **kwargs)
    for prepared in (None, handle):
        ids, weights = flaggems_vllm.mega_gate(x, weight, prepared=prepared, **kwargs)
        torch.testing.assert_close(ids.int(), native_ids)
        torch.testing.assert_close(weights.bfloat16(), native_weights)
    fp32 = dict(kwargs, logits_dtype=torch.float32)
    ids, _ = flaggems_vllm.mega_gate(x, weight, **fp32)
    assert ids[:, -1].tolist() == [2, 1]
    assert native_ids[0, -1].item() == native_ids[1, -1].item()
    with pytest.raises(AssertionError, match="routing parameters"):
        flaggems_vllm.mega_gate(x, weight, prepared=handle, **fp32)


@pytest.mark.parametrize("weight_out", [torch.float32, torch.bfloat16])
def test_mega_gate_direct_outputs(weight_out):
    from flaggems_vllm.runtime.backend._ascend.ops.mega_gate import prepare_mega_gate

    case = GateCase("qwen-35b", 5, 2048, 256, 8, "softmax")
    args = tuple(t.npu() if t is not None else None for t in gate_inputs(case, 2026))
    kwargs = gate_kwargs(case)
    ids, weights = flaggems_vllm.mega_gate(*args, **kwargs)
    kwargs["output_dtypes"] = (torch.int32, weight_out)
    handle = prepare_mega_gate(*args, **kwargs)
    for prepared in (None, handle):
        output = flaggems_vllm.mega_gate(*args, **kwargs, prepared=prepared)
        torch.testing.assert_close(output[0], ids.int(), rtol=0, atol=0)
        torch.testing.assert_close(output[1], weights.to(weight_out), rtol=0, atol=0)
    with pytest.raises(AssertionError, match="routing parameters"):
        flaggems_vllm.mega_gate(*args, prepared=handle, **gate_kwargs(case))


def test_mega_gate_long_prepared_replay():
    from flaggems_vllm.runtime.backend._ascend.ops.mega_gate import prepare_mega_gate

    # Leave a tail tile on just one core pair and reuse the ready flags
    # over several waves, including replay with changed inputs.
    case = GateCase("v32", 16385, 4096, 256, 8, "sigmoid", 8, 4, 2.5)
    kwargs = dict(**gate_kwargs(case), output_dtypes=(torch.int32, torch.float32))
    tensors = tuple(t.npu() for t in gate_inputs(case, 2026))
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        handle = prepare_mega_gate(*tensors, **kwargs)
    validate_graph(
        partial(flaggems_vllm.mega_gate, prepared=handle, **kwargs),
        case,
        gate_inputs,
        partial(check_gate, native=True),
        stream=stream,
    )


@pytest.mark.parametrize(
    "experts,scoring,topk",
    [(384, "sigmoid", 8), (256, "softmax", 8), (512, "softmax", 10)],
)
def test_mega_gate_flat_topk_ties(experts, scoring, topk):
    x = torch.zeros((144, 2048), device="npu", dtype=torch.bfloat16)
    w = x.new_zeros((experts, x.shape[1]))
    bias = (
        x.new_zeros((experts,), dtype=torch.float32) if scoring == "sigmoid" else None
    )
    logits_dtype = torch.float32 if bias is not None else torch.bfloat16
    ids, weights = flaggems_vllm.mega_gate(
        x,
        w,
        bias,
        scoring=scoring,
        topk=topk,
        scale=1.0,
        logits_dtype=logits_dtype,
        output_dtypes=(torch.int32, logits_dtype),
    )
    expected_ids = torch.arange(topk, dtype=torch.int32).expand(x.shape[0], -1)
    torch.testing.assert_close(ids.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        weights.cpu(), torch.full(weights.shape, 1 / topk, dtype=logits_dtype)
    )
