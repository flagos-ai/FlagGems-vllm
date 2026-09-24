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

import ast
import importlib
import inspect

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.FLA import (
    fused_recurrent_kda,
    fused_recurrent_kda_decode,
    fused_recurrent_kda_fwd,
)


def _sm90_available() -> bool:
    if not (torch.cuda.is_available() and flaggems_vllm.device == "cuda"):
        return False
    return torch.cuda.get_device_capability()[0] == 9


requires_sm90 = pytest.mark.skipif(
    not _sm90_available(), reason="optimized recurrent KDA tests require NVIDIA SM90"
)


def _reference_decode(q, k, v, gate, beta, state, state_indices, scale):
    output_dtype = v.dtype
    q = q.float()
    k = k.float()
    v = v.float()
    gate = gate.float()
    beta = beta.float()
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)

    out = torch.zeros_like(v)
    final_state = state.clone()
    _, token_count, _, _ = q.shape
    for token in range(token_count):
        slot = int(state_indices[token].item())
        if slot <= 0:
            continue
        for head in range(4):
            current = final_state[slot, head]
            current = current * torch.exp(gate[0, token, head])[None, :]
            value = v[0, token, head] - (current * k[0, token, head][None, :]).sum(-1)
            value = value * beta[0, token, head]
            current = current + value[:, None] * k[0, token, head][None, :]
            out[0, token, head] = (current * q[0, token, head][None, :]).sum(-1)
            final_state[slot, head] = current
    return out.to(output_dtype), final_state


def _assert_accuracy(actual, expected, *, max_error, relative_rmse):
    actual = actual.float()
    expected = expected.float()
    error = actual - expected
    assert error.abs().max().item() <= max_error
    measured_relative_rmse = (
        error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)
    ).item()
    assert measured_relative_rmse <= relative_rmse


def _build_inputs(token_count, *, padded=False):
    device = flaggems_vllm.device
    torch.manual_seed(1234 + token_count)
    q = torch.randn(1, token_count, 4, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate = -5.0 * torch.rand(1, token_count, 4, 128, dtype=torch.float32, device=device)
    beta = torch.rand(1, token_count, 4, dtype=torch.float32, device=device)
    state = 0.01 * torch.randn(
        token_count + 3, 4, 128, 128, dtype=torch.float32, device=device
    )
    state_indices = torch.arange(1, token_count + 1, dtype=torch.int32, device=device)
    if padded and token_count >= 2:
        state_indices[-2:] = 0
    return q, k, v, gate, beta, state, state_indices


def test_recurrent_kda_public_entries_are_default_closed():
    functions = (
        fused_recurrent_kda_decode,
        fused_recurrent_kda_fwd,
        fused_recurrent_kda,
    )
    for function in functions:
        parameter = inspect.signature(function).parameters["enable_decode_optimization"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is False

    with pytest.raises(NotImplementedError, match="explicitly"):
        fused_recurrent_kda_decode(None, None, None, None, None, None, None)
    with pytest.raises(NotImplementedError, match="explicitly"):
        fused_recurrent_kda_fwd(None, None, None, None, None, 1.0, None)
    with pytest.raises(NotImplementedError, match="explicitly"):
        fused_recurrent_kda(None, None, None, None)
    with pytest.raises(NotImplementedError, match="explicitly"):
        fused_recurrent_kda_decode(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            enable_decode_optimization=1,
        )


def test_recurrent_kda_public_entries_are_not_default_aten_registrations():
    config = dict(flaggems_vllm._FULL_CONFIG)
    functions = {
        "fused_recurrent_kda": fused_recurrent_kda,
        "fused_recurrent_kda_decode": fused_recurrent_kda_decode,
        "fused_recurrent_kda_fwd": fused_recurrent_kda_fwd,
    }
    for name, function in functions.items():
        assert getattr(flaggems_vllm, name) is function
        assert name in flaggems_vllm.ops.__all__
        assert name in flaggems_vllm._EXPLICIT_OPT_IN_ONLY_APIS
        assert name not in config
        assert name not in flaggems_vllm.FULL_CONFIG_BY_FUNC


def test_recurrent_kda_optimized_source_has_no_torch_compute_fallback():
    module = importlib.import_module("flaggems_vllm.ops.FLA.fused_recurrent_kda")
    tree = ast.parse(inspect.getsource(module))
    torch_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id == "torch":
            torch_calls.append(node.func.attr)
    assert torch_calls == ["empty_like"]
    source = inspect.getsource(module)
    assert "@triton.jit" in source
    assert "tilelang" not in source.lower()


def test_recurrent_kda_rejects_output_storage_alias_before_launch():
    q = torch.empty((1, 1, 4, 128), dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    gate = torch.empty((1, 1, 4, 128), dtype=torch.float32)
    beta = torch.empty((1, 1, 4), dtype=torch.float32)
    state = torch.empty((2, 4, 128, 128), dtype=torch.float32)
    slots = torch.ones((1,), dtype=torch.int32)

    with pytest.raises(ValueError, match="must not alias"):
        fused_recurrent_kda_decode(
            q,
            k,
            v,
            gate,
            beta,
            state,
            slots,
            out=q,
            enable_decode_optimization=True,
        )


@pytest.mark.fused_recurrent_kda
@requires_sm90
@pytest.mark.parametrize("token_count", [1, 32, 64, 96, 128])
@torch.inference_mode()
def test_fused_recurrent_kda_decode_matches_reference(token_count):
    q, k, v, gate, beta, state, slots = _build_inputs(token_count)
    expected_out, expected_state = _reference_decode(
        q, k, v, gate, beta, state, slots, 128**-0.5
    )

    actual_state = state.clone()
    actual_out, final_state = fused_recurrent_kda_decode(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=actual_state,
        ssm_state_indices=slots,
        enable_decode_optimization=True,
    )

    assert final_state.data_ptr() == actual_state.data_ptr()
    _assert_accuracy(actual_out, expected_out, max_error=3.2e-2, relative_rmse=5e-3)
    _assert_accuracy(actual_state, expected_state, max_error=1e-4, relative_rmse=1e-4)


@pytest.mark.fused_recurrent_kda
@requires_sm90
@torch.inference_mode()
def test_fused_recurrent_kda_null_block_id_does_not_touch_state():
    q, k, v, gate, beta, state, slots = _build_inputs(8, padded=True)
    expected_out, expected_state = _reference_decode(
        q, k, v, gate, beta, state, slots, 128**-0.5
    )
    initial_slot_zero = state[0].clone()

    actual_out, _ = fused_recurrent_kda_decode(
        q,
        k,
        v,
        gate,
        beta,
        state,
        slots,
        enable_decode_optimization=True,
    )

    _assert_accuracy(actual_out, expected_out, max_error=3.2e-2, relative_rmse=5e-3)
    _assert_accuracy(state, expected_state, max_error=1e-4, relative_rmse=1e-4)
    assert torch.equal(state[0], initial_slot_zero)
    assert torch.count_nonzero(actual_out[0, -2:]).item() == 0


@pytest.mark.fused_recurrent_kda
@requires_sm90
@torch.inference_mode()
def test_fused_recurrent_kda_fwd_matches_vllm_decode_abi():
    q, k, v, gate, beta, state, slots = _build_inputs(32)
    cu_seqlens = torch.arange(33, dtype=torch.int32, device=q.device)
    expected_out, expected_state = _reference_decode(
        q, k, v, gate, beta, state, slots, 128**-0.5
    )

    actual_state = state.clone()
    actual_out, final_state = fused_recurrent_kda_fwd(
        q,
        k,
        v,
        gate,
        beta,
        128**-0.5,
        actual_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=slots,
        use_qk_l2norm_in_kernel=True,
        enable_decode_optimization=True,
    )

    assert final_state.data_ptr() == actual_state.data_ptr()
    _assert_accuracy(actual_out, expected_out, max_error=3.2e-2, relative_rmse=5e-3)
    _assert_accuracy(actual_state, expected_state, max_error=1e-4, relative_rmse=1e-4)


@pytest.mark.fused_recurrent_kda
@requires_sm90
@torch.inference_mode()
def test_fused_recurrent_kda_accepts_trailing_graph_padding():
    active_count, graph_batch = 6, 8
    q, k, v, gate, beta, state, active_slots = _build_inputs(active_count)
    slots = torch.cat(
        (
            active_slots,
            torch.zeros(graph_batch - active_count, dtype=torch.int32, device=q.device),
        )
    )
    cu_seqlens = torch.cat(
        (
            torch.arange(active_count + 1, dtype=torch.int32, device=q.device),
            torch.full(
                (graph_batch - active_count,),
                active_count,
                dtype=torch.int32,
                device=q.device,
            ),
        )
    )
    expected_out, expected_state = _reference_decode(
        q, k, v, gate, beta, state, active_slots, 128**-0.5
    )

    actual_state = state.clone()
    actual_out, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=actual_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=slots,
        enable_decode_optimization=True,
    )

    _assert_accuracy(actual_out, expected_out, max_error=3.2e-2, relative_rmse=5e-3)
    _assert_accuracy(actual_state, expected_state, max_error=1e-4, relative_rmse=1e-4)
    assert torch.equal(actual_state[0], state[0])


@pytest.mark.fused_recurrent_kda
@requires_sm90
@torch.inference_mode()
def test_fused_recurrent_kda_cuda_graph_replay_preserves_null_slot():
    q, k, v, gate, beta, state, slots = _build_inputs(32, padded=True)
    cu_seqlens = torch.arange(33, dtype=torch.int32, device=q.device)
    cu_seqlens[-2:] = 30
    expected_out, expected_state = _reference_decode(
        q, k, v, gate, beta, state, slots, 128**-0.5
    )

    warmup_state = state.clone()
    warmup_out = torch.empty_like(v)
    fused_recurrent_kda_decode(
        q,
        k,
        v,
        gate,
        beta,
        warmup_state,
        slots,
        enable_decode_optimization=True,
        cu_seqlens=cu_seqlens,
        out=warmup_out,
    )
    torch.cuda.synchronize()

    actual_state = state.clone()
    actual_out = torch.empty_like(v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned_out, final_state = fused_recurrent_kda_decode(
            q,
            k,
            v,
            gate,
            beta,
            actual_state,
            slots,
            enable_decode_optimization=True,
            cu_seqlens=cu_seqlens,
            out=actual_out,
        )
    actual_state.copy_(state)
    graph.replay()
    torch.cuda.synchronize()

    assert returned_out.data_ptr() == actual_out.data_ptr()
    assert final_state.data_ptr() == actual_state.data_ptr()
    _assert_accuracy(actual_out, expected_out, max_error=3.2e-2, relative_rmse=5e-3)
    _assert_accuracy(actual_state, expected_state, max_error=1e-4, relative_rmse=1e-4)
    assert torch.equal(actual_state[0], state[0])
    assert torch.count_nonzero(actual_out[0, -2:]).item() == 0


@pytest.mark.fused_recurrent_kda
@requires_sm90
def test_fused_recurrent_kda_rejects_non_target_cases():
    q, k, v, gate, beta, state, slots = _build_inputs(2)

    with pytest.raises(ValueError, match="q shape"):
        fused_recurrent_kda_decode(
            q[:, :, :2],
            k[:, :, :2],
            v[:, :, :2],
            gate[:, :, :2],
            beta[:, :, :2],
            state[:, :2],
            slots,
            enable_decode_optimization=True,
        )
    with pytest.raises(ValueError, match="Speculative decode"):
        fused_recurrent_kda_fwd(
            q,
            k,
            v,
            gate,
            beta,
            128**-0.5,
            state,
            ssm_state_indices=slots,
            num_accepted_tokens=slots,
            use_qk_l2norm_in_kernel=True,
            enable_decode_optimization=True,
        )
    with pytest.raises(ValueError, match="at least one sequence|token_capacity"):
        fused_recurrent_kda_fwd(
            q,
            k,
            v,
            gate,
            beta,
            128**-0.5,
            state,
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32, device=q.device),
            ssm_state_indices=torch.ones(1, dtype=torch.int32, device=q.device),
            use_qk_l2norm_in_kernel=True,
            enable_decode_optimization=True,
        )
