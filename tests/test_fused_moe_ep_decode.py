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
import inspect
from pathlib import Path

import pytest
import torch

from flaggems_vllm.ops import fused_moe
from flaggems_vllm.ops.fused_moe_ep_decode import (
    _fused_moe_ep_decode,
    _prepare_ep_decode_buffer,
    _select_ep_decode_plan,
    _tensors_overlap,
)


def _has_h20() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (
        9,
        0,
    ) and "H20" in torch.cuda.get_device_name().replace(" ", "_").split("_")


def test_ep_decode_selector_is_explicit_and_exact():
    kwargs = {
        "enabled": True,
        "is_h20": True,
        "num_tokens": 96,
        "local_num_experts": 18,
        "global_num_experts": 288,
        "top_k": 8,
        "hidden_size": 4096,
        "intermediate_size": 2048,
        "clamp_limit": 10.0,
        "dtype": torch.bfloat16,
        "has_expert_map": True,
    }
    assert _select_ep_decode_plan(**kwargs) == "compact_routed"
    assert (
        _select_ep_decode_plan(**{**kwargs, "num_tokens": 1}) == "adaptive_single_token"
    )
    assert (
        _select_ep_decode_plan(**{**kwargs, "num_tokens": 1, "intermediate_size": 1280})
        == "direct_single_token"
    )

    for key, value in (
        ("enabled", False),
        ("enabled", 1),
        ("is_h20", False),
        ("num_tokens", 129),
        ("local_num_experts", 17),
        ("global_num_experts", 256),
        ("top_k", 4),
        ("hidden_size", 6144),
        ("intermediate_size", 4096),
        ("clamp_limit", None),
        ("clamp_limit", 7.0),
        ("dtype", torch.float16),
        ("has_expert_map", False),
    ):
        assert _select_ep_decode_plan(**{**kwargs, key: value}) is None


@pytest.mark.parametrize(
    "entrypoint",
    [
        fused_moe.fused_experts_impl,
        fused_moe.inplace_fused_experts,
        fused_moe.outplace_fused_experts,
    ],
)
def test_fused_experts_opt_in_parameter_is_keyword_only_and_default_closed(entrypoint):
    parameters = inspect.signature(entrypoint).parameters
    flag = parameters["enable_ep_decode_optimization"]
    clamp = parameters["gemm1_clamp_limit"]
    assert flag.kind is inspect.Parameter.KEYWORD_ONLY
    assert flag.default is False
    assert clamp.kind is inspect.Parameter.KEYWORD_ONLY
    assert clamp.default is None
    for name in ("intermediate_cache13", "intermediate_cache2"):
        parameter = parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None
    if entrypoint is not fused_moe.inplace_fused_experts:
        output = parameters["output"]
        assert output.kind is inspect.Parameter.KEYWORD_ONLY
        assert output.default is None
    if entrypoint is not fused_moe.fused_experts_impl:
        expert_map = parameters["expert_map"]
        assert expert_map.kind is inspect.Parameter.KEYWORD_ONLY
        assert expert_map.default is None


def test_fused_experts_wrappers_forward_opt_in_keywords(monkeypatch):
    marker = object()
    calls = []

    def fake_impl(*args, **kwargs):
        calls.append((args, kwargs))
        return marker

    monkeypatch.setattr(fused_moe, "fused_experts_impl", fake_impl)
    tensor = torch.empty((0, 0))
    expert_map = torch.empty((0,), dtype=torch.int32)
    cache13 = torch.empty((0,))
    cache2 = torch.empty((0,))
    output = torch.empty((0, 0))
    result = fused_moe.inplace_fused_experts(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        expert_map=expert_map,
        gemm1_clamp_limit=10.0,
        enable_ep_decode_optimization=True,
        intermediate_cache13=cache13,
        intermediate_cache2=cache2,
    )
    assert result is None
    result = fused_moe.outplace_fused_experts(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        expert_map=expert_map,
        gemm1_clamp_limit=10.0,
        enable_ep_decode_optimization=True,
        output=output,
        intermediate_cache13=cache13,
        intermediate_cache2=cache2,
    )
    assert result is marker
    assert [call[1]["inplace"] for call in calls] == [True, False]
    for _, kwargs in calls:
        assert kwargs["expert_map"] is expert_map
        assert kwargs["gemm1_clamp_limit"] == 10.0
        assert kwargs["enable_ep_decode_optimization"] is True
        assert kwargs["intermediate_cache13"] is cache13
        assert kwargs["intermediate_cache2"] is cache2
    assert "output" not in calls[0][1]
    assert calls[1][1]["output"] is output


def test_fused_experts_dispatches_only_when_enabled(monkeypatch):
    marker = object()
    calls = []

    def fake_optimized(*args, **kwargs):
        calls.append((args, kwargs))
        return marker

    monkeypatch.setattr(fused_moe, "_fused_moe_ep_decode", fake_optimized)
    tensor = torch.empty((0, 0))
    output = torch.empty((0, 0))
    cache13 = torch.empty((0,))
    cache2 = torch.empty((0,))
    result = fused_moe.fused_experts_impl(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        global_num_experts=288,
        gemm1_clamp_limit=10.0,
        enable_ep_decode_optimization=True,
        output=output,
        intermediate_cache13=cache13,
        intermediate_cache2=cache2,
    )
    assert result is marker
    assert len(calls) == 1

    with pytest.raises(TypeError, match="must be a bool"):
        fused_moe.fused_experts_impl(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            enable_ep_decode_optimization=1,
        )
    assert len(calls) == 1
    assert calls[0][1]["gemm1_clamp_limit"] == 10.0
    assert calls[0][1]["output"] is output
    assert calls[0][1]["intermediate_cache13"] is cache13
    assert calls[0][1]["intermediate_cache2"] is cache2

    with pytest.raises(AssertionError, match="Only 'silu'"):
        fused_moe.fused_experts_impl(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            activation="unsupported",
        )
    assert len(calls) == 1


def test_clamp_cannot_silently_use_legacy_unclamped_silu():
    tensor = torch.empty((0, 0))
    with pytest.raises(NotImplementedError, match="must not silently"):
        fused_moe.fused_experts_impl(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            gemm1_clamp_limit=10.0,
        )


@pytest.mark.parametrize(
    "buffer_name",
    ["output", "intermediate_cache13", "intermediate_cache2"],
)
def test_caller_owned_buffers_cannot_reach_default_path(buffer_name):
    tensor = torch.empty((0, 0))
    kwargs = {buffer_name: tensor}
    with pytest.raises(NotImplementedError, match="require.*optimization=True"):
        fused_moe.fused_experts_impl(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            **kwargs,
        )


def test_ep_decode_buffer_helpers_validate_and_slice_caller_storage():
    reference = torch.empty((2, 4), dtype=torch.bfloat16)
    storage = torch.empty((20,), dtype=torch.bfloat16)
    prepared = _prepare_ep_decode_buffer(
        storage,
        name="workspace",
        reference=reference,
        required_numel=12,
    )
    assert prepared.numel() == 12
    assert prepared.data_ptr() == storage.data_ptr()
    assert _tensors_overlap(prepared, storage[11:])
    assert not _tensors_overlap(prepared, storage[12:])

    with pytest.raises(ValueError, match="too small"):
        _prepare_ep_decode_buffer(
            storage[:11],
            name="workspace",
            reference=reference,
            required_numel=12,
        )
    with pytest.raises(ValueError, match="dtype"):
        _prepare_ep_decode_buffer(
            torch.empty((12,), dtype=torch.float32),
            name="workspace",
            reference=reference,
            required_numel=12,
        )
    with pytest.raises(ValueError, match="contiguous"):
        _prepare_ep_decode_buffer(
            storage[:24:2],
            name="workspace",
            reference=reference,
            required_numel=10,
        )


def _call_host_ep_decode(monkeypatch, *, output, cache13, cache2, inplace=False):
    module = importlib.import_module("flaggems_vllm.ops.fused_moe_ep_decode")
    hidden = torch.empty((1, 16), dtype=torch.bfloat16)
    w1 = torch.empty((1, 8, 16), dtype=torch.bfloat16)
    w2 = torch.empty((1, 16, 4), dtype=torch.bfloat16)
    topk_weights = torch.empty((1, 8), dtype=torch.float32)
    topk_ids = torch.empty((1, 8), dtype=torch.int32)
    expert_map = torch.empty((288,), dtype=torch.int32)

    monkeypatch.setattr(
        module, "_validate_inputs", lambda *args, **kwargs: "adaptive_single_token"
    )

    def fake_local_rank(*args, **kwargs):
        return args[8]

    monkeypatch.setattr(module, "_fused_moe_ep_m1_i2048_local_rank", fake_local_rank)
    result = _fused_moe_ep_decode(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=inplace,
        activation="silu",
        apply_router_weight_on_input=False,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        ocp_mx_scheme=None,
        per_channel_quant=False,
        global_num_experts=288,
        expert_map=expert_map,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        w1_bias=None,
        w2_bias=None,
        gemm1_clamp_limit=10.0,
        output=output,
        intermediate_cache13=cache13,
        intermediate_cache2=cache2,
    )
    return result, hidden


def test_ep_decode_accepts_output_cache2_alias_and_preserves_output_identity(
    monkeypatch,
):
    cache13 = torch.empty((8 * 4096,), dtype=torch.bfloat16)
    cache2 = torch.empty((8 * 4,), dtype=torch.bfloat16)
    output = cache2[:16].view(1, 16)
    result, _ = _call_host_ep_decode(
        monkeypatch,
        output=output,
        cache13=cache13,
        cache2=cache2,
    )
    assert result is output


def test_ep_decode_rejects_unsafe_caller_buffer_aliases(monkeypatch):
    shared = torch.empty((8 * 4096,), dtype=torch.bfloat16)
    output = torch.empty((1, 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="cache13 and intermediate_cache2"):
        _call_host_ep_decode(
            monkeypatch,
            output=output,
            cache13=shared,
            cache2=shared,
        )

    cache2 = torch.empty((8 * 4,), dtype=torch.bfloat16)
    output = shared[:16].view(1, 16)
    with pytest.raises(
        ValueError, match="output must not overlap intermediate_cache13"
    ):
        _call_host_ep_decode(
            monkeypatch,
            output=output,
            cache13=shared,
            cache2=cache2,
        )


def test_ep_decode_rejects_inplace_with_output_before_device_dispatch(monkeypatch):
    with pytest.raises(ValueError, match="inplace=True and output"):
        _call_host_ep_decode(
            monkeypatch,
            output=torch.empty((1, 16), dtype=torch.bfloat16),
            cache13=None,
            cache2=None,
            inplace=True,
        )


@pytest.mark.parametrize(
    "grad_input",
    ["hidden", "w1", "w2", "topk_weights", "topk_ids", "expert_map"],
)
def test_optimized_path_rejects_autograd_inputs_before_launch(grad_input):
    tensors = {
        "hidden": torch.empty((1, 1)),
        "w1": torch.empty((1, 1, 1)),
        "w2": torch.empty((1, 1, 1)),
        "topk_weights": torch.empty((1, 1)),
        "topk_ids": torch.empty((1, 1), dtype=torch.int64),
        "expert_map": torch.empty((1,), dtype=torch.int32),
    }
    if grad_input in ("topk_ids", "expert_map"):
        # Integer tensors cannot require gradients. A floating substitute proves
        # the inference-only guard runs before the optimized dtype validation.
        tensors[grad_input] = torch.empty_like(
            tensors[grad_input], dtype=torch.float32, requires_grad=True
        )
    else:
        tensors[grad_input].requires_grad_(True)

    with pytest.raises(NotImplementedError, match="inference-only"):
        fused_moe.fused_experts_impl(
            tensors["hidden"],
            tensors["w1"],
            tensors["w2"],
            tensors["topk_weights"],
            tensors["topk_ids"],
            global_num_experts=288,
            expert_map=tensors["expert_map"],
            gemm1_clamp_limit=10.0,
            enable_ep_decode_optimization=True,
        )


def test_optimized_production_files_have_no_non_triton_gpu_compute():
    ops_dir = Path(fused_moe.__file__).parent
    sources = [
        (ops_dir / "fused_moe_ep_decode.py").read_text(),
        (ops_dir / "fused_moe_ep_m1.py").read_text(),
    ]
    moe_sum_module = importlib.import_module("flaggems_vllm.ops.moe_sum")
    sources.append(moe_sum_module._moe_sum_ep_kernel.src)
    sources.append(inspect.getsource(moe_sum_module._moe_sum_ep))
    forbidden = (
        "torch.mm(",
        "torch.matmul(",
        "torch.ops.",
        "torch.nn.functional",
        ".zero_(",
        ".fill_(",
        ".copy_(",
        ".item(",
    )
    for source in sources:
        assert "tle." not in source
        for token in forbidden:
            assert token not in source


def _reference_ep_rank(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    expert_map,
):
    num_tokens, hidden_size = hidden_states.shape
    output = torch.zeros(
        (num_tokens, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for token in range(num_tokens):
        for route in range(topk_ids.shape[1]):
            global_expert = int(topk_ids[token, route].item())
            if global_expert < 0 or global_expert >= expert_map.numel():
                continue
            local_expert = int(expert_map[global_expert].item())
            if local_expert < 0 or local_expert >= w1.shape[0]:
                continue
            projection = torch.matmul(
                hidden_states[token].float(),
                w1[local_expert].transpose(0, 1).float(),
            ).to(torch.bfloat16)
            gate, up = projection.float().chunk(2)
            gate = gate.clamp(max=10.0)
            up = up.clamp(min=-10.0, max=10.0)
            activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            routed = torch.matmul(
                activated.float(),
                w2[local_expert].transpose(0, 1).float(),
            )
            routed = (routed * topk_weights[token, route].float()).to(torch.bfloat16)
            output[token] += routed.float()
    return output.to(torch.bfloat16)


@pytest.mark.parametrize("intermediate_size", [1280, 2048])
@pytest.mark.parametrize("num_tokens", [1, 32, 64, 96, 128])
@pytest.mark.skipif(not _has_h20(), reason="exact optimized path requires NVIDIA H20")
def test_ep_decode_matches_bf16_boundary_reference(num_tokens, intermediate_size):
    torch.manual_seed(7)
    device = torch.device("cuda")
    hidden_states = torch.randn((num_tokens, 4096), device=device, dtype=torch.bfloat16)
    w1 = (
        torch.randn(
            (18, 2 * intermediate_size, 4096),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.01
    )
    w2 = (
        torch.randn(
            (18, 4096, intermediate_size),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.01
    )
    expert_map = torch.full((288,), -1, device=device, dtype=torch.int32)
    local_begin = 7 * 18
    expert_map[local_begin : local_begin + 18] = torch.arange(
        18, device=device, dtype=torch.int32
    )

    # Cover no-local, early/late singleton, multiple local, and invalid int64
    # route IDs. Invalid IDs are checked before expert_map addressing.
    topk_ids = torch.zeros((num_tokens, 8), device=device, dtype=torch.int64)
    if num_tokens >= 1:
        topk_ids[0] = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, local_begin],
            device=device,
            dtype=torch.int64,
        )
    if num_tokens >= 2:
        topk_ids[1, 0] = local_begin
    if num_tokens >= 3:
        topk_ids[2, 7] = local_begin + 1
    if num_tokens >= 4:
        topk_ids[3, 1] = local_begin + 2
        topk_ids[3, 6] = local_begin + 3
    if num_tokens >= 5:
        topk_ids[4, 0] = 2**40
        topk_ids[4, 1] = -(2**40)
    for token in range(5, num_tokens):
        if token % 16 == 0:
            topk_ids[token, token % 8] = local_begin + token % 18

    topk_weights = torch.rand((num_tokens, 8), device=device, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    expected = _reference_ep_rank(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        expert_map,
    )
    inplace = num_tokens == 1 and intermediate_size == 2048
    input_storage = hidden_states.clone() if inplace else hidden_states
    cache13 = torch.empty(
        num_tokens * 8 * 4096,
        device=device,
        dtype=torch.bfloat16,
    )
    cache2_storage = torch.empty(
        max(num_tokens * 8 * intermediate_size, num_tokens * 4096),
        device=device,
        dtype=torch.bfloat16,
    )
    caller_output = (
        None if inplace else cache2_storage[: num_tokens * 4096].view_as(hidden_states)
    )
    actual = fused_moe.fused_experts_impl(
        input_storage,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=inplace,
        global_num_experts=288,
        expert_map=expert_map,
        gemm1_clamp_limit=10.0,
        enable_ep_decode_optimization=True,
        output=caller_output,
        intermediate_cache13=cache13,
        intermediate_cache2=cache2_storage,
    )
    if inplace:
        assert actual.data_ptr() == input_storage.data_ptr()
    else:
        assert actual is caller_output
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)

    if inplace:
        input_storage.copy_(hidden_states)
    cache13.fill_(float("nan"))
    cache2_storage.fill_(float("nan"))
    repeated = fused_moe.fused_experts_impl(
        input_storage,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=inplace,
        global_num_experts=288,
        expert_map=expert_map,
        gemm1_clamp_limit=10.0,
        enable_ep_decode_optimization=True,
        output=caller_output,
        intermediate_cache13=cache13,
        intermediate_cache2=cache2_storage,
    )
    torch.testing.assert_close(repeated, expected, rtol=3e-2, atol=2e-2)


@pytest.mark.parametrize(
    "route_case,id_dtype,map_dtype,inplace",
    [
        ("all_remote", torch.int32, torch.int32, False),
        ("all_local", torch.int64, torch.int64, False),
        ("skewed_local", torch.int32, torch.int64, True),
        ("invalid_ids", torch.int64, torch.int32, True),
    ],
)
@pytest.mark.skipif(not _has_h20(), reason="exact optimized path requires NVIDIA H20")
def test_ep_decode_route_extremes_and_aliasing(
    route_case, id_dtype, map_dtype, inplace
):
    torch.manual_seed(11)
    device = torch.device("cuda")
    num_tokens, intermediate_size = 32, 1280
    hidden_states = torch.randn((num_tokens, 4096), device=device, dtype=torch.bfloat16)
    w1 = (
        torch.randn(
            (18, 2 * intermediate_size, 4096),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.01
    )
    w2 = (
        torch.randn(
            (18, 4096, intermediate_size),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.01
    )
    local_begin = 7 * 18
    expert_map = torch.full((288,), -1, device=device, dtype=map_dtype)
    expert_map[local_begin : local_begin + 18] = torch.arange(
        18, device=device, dtype=map_dtype
    )
    if route_case == "all_remote":
        route_row = torch.arange(8, device=device, dtype=id_dtype)
    elif route_case == "all_local":
        route_row = torch.arange(
            local_begin, local_begin + 8, device=device, dtype=id_dtype
        )
    elif route_case == "skewed_local":
        route_row = torch.tensor(
            [local_begin, 0, 1, 2, 3, 4, 5, 6],
            device=device,
            dtype=id_dtype,
        )
    else:
        route_row = torch.tensor(
            [2**40, -(2**40), local_begin, 0, 1, 2, 3, 4],
            device=device,
            dtype=id_dtype,
        )
    topk_ids = route_row.repeat(num_tokens, 1)
    weight_dtype = torch.bfloat16 if route_case == "all_local" else torch.float32
    topk_weights = torch.rand((num_tokens, 8), device=device, dtype=weight_dtype)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    expected = _reference_ep_rank(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        expert_map,
    )
    input_storage = hidden_states.clone() if inplace else hidden_states
    actual = fused_moe.fused_experts_impl(
        input_storage,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=inplace,
        global_num_experts=288,
        expert_map=expert_map,
        gemm1_clamp_limit=10.0,
        enable_ep_decode_optimization=True,
    )
    if inplace:
        assert actual.data_ptr() == input_storage.data_ptr()
    else:
        assert actual.data_ptr() != hidden_states.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)
    if route_case == "all_remote":
        assert torch.count_nonzero(actual) == 0


@pytest.mark.skipif(not _has_h20(), reason="exact optimized path requires NVIDIA H20")
def test_ep_m1_i2048_cuda_graph_replays_dynamic_routes_and_map():
    torch.manual_seed(19)
    device = torch.device("cuda")
    hidden_states = torch.randn((1, 4096), device=device, dtype=torch.bfloat16)
    w1 = torch.randn((18, 4096, 4096), device=device, dtype=torch.bfloat16) * 0.01
    w2 = torch.randn((18, 4096, 2048), device=device, dtype=torch.bfloat16) * 0.01
    topk_weights = torch.rand((1, 8), device=device, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    topk_ids = torch.arange(8, device=device, dtype=torch.int64).view(1, 8)
    expert_map = torch.full((288,), -1, device=device, dtype=torch.int64)
    local_begin = 7 * 18
    expert_map[local_begin : local_begin + 18] = torch.arange(
        18, device=device, dtype=torch.int64
    )
    cache13 = torch.empty((8 * 4096,), device=device, dtype=torch.bfloat16)
    cache2_storage = torch.empty((8 * 2048,), device=device, dtype=torch.bfloat16)
    graph_output_storage = cache2_storage[:4096].view_as(hidden_states)

    def optimized():
        return fused_moe.fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            global_num_experts=288,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            enable_ep_decode_optimization=True,
            output=graph_output_storage,
            intermediate_cache13=cache13,
            intermediate_cache2=cache2_storage,
        )

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            graph_output = optimized()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = optimized()

    route_cases = [
        torch.arange(8, device=device, dtype=torch.int64),
        torch.tensor(
            [local_begin, 0, 1, 2, 3, 4, 5, 6],
            device=device,
            dtype=torch.int64,
        ),
        torch.tensor(
            [0, local_begin, 1, local_begin + 1, 2, 3, 4, 5],
            device=device,
            dtype=torch.int64,
        ),
    ]
    for route_row in route_cases:
        topk_ids.copy_(route_row.view(1, 8))
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference_ep_rank(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            expert_map,
        )
        torch.testing.assert_close(graph_output, expected, rtol=3e-2, atol=2e-2)

    # Change both the map contents and route IDs without recapturing. The
    # captured kernels must reload device data and choose the new local ranks.
    shifted_map = torch.full_like(expert_map, -1)
    shifted_begin = 6 * 18
    shifted_map[shifted_begin : shifted_begin + 18] = torch.arange(
        18, device=device, dtype=torch.int64
    )
    expert_map.copy_(shifted_map)
    topk_ids.copy_(
        torch.tensor(
            [[shifted_begin, 0, shifted_begin + 1, 1, 2, 3, 4, 5]],
            device=device,
            dtype=torch.int64,
        )
    )
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference_ep_rank(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        expert_map,
    )
    torch.testing.assert_close(graph_output, expected, rtol=3e-2, atol=2e-2)
