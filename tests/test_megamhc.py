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

"""Tests for the public Hopper MegaMHC entry points."""

from __future__ import annotations

import builtins
import threading
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamhc import (
    DEEPSEEK_V4_FLASH_PROFILE,
    MegaMHCContext,
    MegaMHCTensorBinding,
    MegaMHCWorkspace,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamhc import api as megamhc_api
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamhc import (
    factory as megamhc_factory,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamhc import mega_mhc
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamhc.factory import (
    _create_context_from_prepared,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamhc.tle import (
    prepared_launcher as prepared_launcher_module,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamhc.tle.dispatch import (
    EAGER_TOKENS,
    GRAPH_TOKENS,
)
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

pytestmark = pytest.mark.megamhc


def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.megamhc
@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
def test_megamhc_is_exported_on_hopper():
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops import (
        megamhc as hopper_megamhc,
    )

    assert flaggems_vllm.create_mega_mhc_context is (
        hopper_megamhc.create_mega_mhc_context
    )
    assert flaggems_vllm.mega_mhc is hopper_megamhc.mega_mhc


def _make_inputs(tokens: int):
    generator = torch.Generator(device="cuda").manual_seed(20261111 + tokens)

    def randn(shape, dtype):
        return torch.randn(shape, dtype=dtype, device="cuda", generator=generator)

    comb = torch.rand((tokens, 4, 4), device="cuda", generator=generator)
    for _ in range(3):
        comb /= comb.sum(-1, keepdim=True)
        comb /= comb.sum(-2, keepdim=True)
    return {
        "x": randn((tokens, 4096), torch.bfloat16),
        "residual": randn((tokens, 4, 4096), torch.bfloat16),
        "post": randn((tokens, 4, 1), torch.float32).sigmoid(),
        "comb": comb,
        "fn": randn((24, 16384), torch.float32).mul_(1e-2),
        "scale": randn((3,), torch.float32).mul_(0.1),
        "base": randn((24,), torch.float32).mul_(0.1),
        "norm": randn((4096,), torch.float32).mul_(0.1).add_(1).bfloat16(),
    }


def _reference(inputs):
    x = inputs["x"]
    residual = inputs["residual"]
    post = inputs["post"]
    new_residual = x.float().unsqueeze(1) * post
    new_residual += torch.einsum("tij,tih->tjh", inputs["comb"], residual.float())
    new_residual = new_residual.bfloat16()
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        mixes = new_residual.flatten(1).float() @ inputs["fn"].mT
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    mixes *= torch.rsqrt(
        new_residual.float().square().sum((1, 2)) / (4 * 4096) + 1e-6
    ).unsqueeze(1)
    scales = torch.cat(
        (
            inputs["scale"][0].expand(4),
            inputs["scale"][1].expand(4),
            inputs["scale"][2].expand(16),
        )
    )
    mixes = mixes * scales + inputs["base"]
    pre = mixes[:, :4].sigmoid().unsqueeze(2) + 1e-6
    new_post = (mixes[:, 4:8].sigmoid() * 2.0).unsqueeze(2)
    new_comb = mixes[:, 8:].view(-1, 4, 4).softmax(dim=-1) + 1e-6
    new_comb /= new_comb.sum(dim=-2, keepdim=True) + 1e-6
    for _ in range(19):
        new_comb /= new_comb.sum(dim=-1, keepdim=True) + 1e-6
        new_comb /= new_comb.sum(dim=-2, keepdim=True) + 1e-6
    norm_input = (new_residual.float() * pre).sum(dim=1).bfloat16().float()
    y = (
        norm_input
        * torch.rsqrt(norm_input.square().mean(1) + 1e-6).unsqueeze(1)
        * inputs["norm"].float()
    ).bfloat16()
    return new_residual, new_post, new_comb, y


def _relative_diff(actual, expected):
    actual = actual.double()
    expected = expected.double()
    denominator = (actual.square() + expected.square()).sum()
    if float(denominator) == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (actual * expected).sum() / denominator)


def _run_gpu_case(inputs, context):
    return flaggems_vllm.mega_mhc(
        inputs["x"],
        inputs["residual"],
        inputs["post"],
        inputs["comb"],
        inputs["fn"],
        inputs["scale"],
        inputs["base"],
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        1,
        1,
        inputs["norm"],
        1e-6,
        context=context,
    )


def _assert_gpu_outputs(actual, expected):
    for result, reference in zip(actual, expected, strict=True):
        assert result.shape == reference.shape
        assert result.dtype == reference.dtype
        assert torch.isfinite(result).all()
        assert _relative_diff(result, reference) < 5e-5


@pytest.mark.megamhc
@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
def test_megamhc_mixed_token_contexts_are_accurate():
    cases = []
    for tokens in (1, 64, 512):
        inputs = _make_inputs(tokens)
        context = flaggems_vllm.create_mega_mhc_context(
            inputs["x"],
            inputs["residual"],
            inputs["post"],
            inputs["comb"],
            inputs["fn"],
            inputs["scale"],
            inputs["base"],
            inputs["norm"],
        )
        assert context.supported_tokens == frozenset({tokens})
        cases.append((inputs, context, _reference(inputs)))

    for inputs, context, expected in (*reversed(cases), *cases):
        actual = _run_gpu_case(inputs, context)
        torch.cuda.synchronize()
        _assert_gpu_outputs(actual, expected)


@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
@pytest.mark.parametrize("tokens", (*GRAPH_TOKENS, *EAGER_TOKENS))
def test_megamhc_production_matrix(tokens, pytestconfig):
    if pytestconfig.getoption("--quick", default=False):
        pytest.skip("full dispatch matrix requires a non-quick run")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        inputs = _make_inputs(tokens)
        context = flaggems_vllm.create_mega_mhc_context(*inputs.values())
        _assert_gpu_outputs(_run_gpu_case(inputs, context), _reference(inputs))

        if tokens in GRAPH_TOKENS:
            from cuda.bindings import driver

            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                actual = _run_gpu_case(inputs, context)
            handle = driver.CUgraph(graph.raw_cuda_graph())
            status, _, node_count = driver.cuGraphGetNodes(handle)
            assert status == driver.CUresult.CUDA_SUCCESS
            assert node_count == 1
            status, (node,), _ = driver.cuGraphGetNodes(handle, node_count)
            assert status == driver.CUresult.CUDA_SUCCESS
            status, node_type = driver.cuGraphNodeGetType(node)
            assert status == driver.CUresult.CUDA_SUCCESS
            assert node_type == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
            for _ in range(2):
                inputs["x"].neg_()
                graph.replay()
                _assert_gpu_outputs(actual, _reference(inputs))
        else:
            for _ in range(2):
                inputs["x"].neg_()
                _assert_gpu_outputs(_run_gpu_case(inputs, context), _reference(inputs))


@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
def test_megamhc_independent_eager_streams(pytestconfig):
    if pytestconfig.getoption("--quick", default=False):
        pytest.skip("independent stream coverage requires a non-quick run")

    cases = []
    for tokens in (1028, 2048):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            inputs = _make_inputs(tokens)
            context = flaggems_vllm.create_mega_mhc_context(*inputs.values())
        cases.append((stream, inputs, context))

    outputs = []
    for stream, inputs, context in reversed(cases):
        with torch.cuda.stream(stream):
            outputs.append(_run_gpu_case(inputs, context))

    for (stream, inputs, _), actual in zip(reversed(cases), outputs, strict=True):
        with torch.cuda.stream(stream):
            _assert_gpu_outputs(actual, _reference(inputs))


# Host-side context, binding, workspace, and fail-closed contract tests.


class _RecordingLauncher:
    def __init__(
        self,
        *,
        supported_tokens=frozenset({1}),
        device=torch.device("cpu"),
        stream_handle=None,
        dynamic_alignments=None,
        static_alignments=(16, 16, 16, 16),
    ):
        self.calls = []
        self.supported_tokens = supported_tokens
        self.device = device
        self.stream_handle = stream_handle
        self.owner_thread_id = threading.get_ident()
        self._dynamic_alignments = {token: (16,) * 8 for token in supported_tokens}
        if dynamic_alignments is not None:
            self._dynamic_alignments.update(dynamic_alignments)
        self.static_alignments = static_alignments

    def dynamic_alignments(self, tokens):
        return self._dynamic_alignments[tokens]

    def launch(self, *args):
        self.calls.append(args)


class _RaisingLauncher(_RecordingLauncher):
    def launch(self, *args):
        super().launch(*args)
        raise RuntimeError("synthetic launch failure")


class _PreparedLauncher:
    def __init__(
        self,
        token_count,
        static_bound_values,
        *,
        dynamic_alignments=(16,) * 8,
        static_alignments=(16,) * 4,
        device=torch.device("cpu"),
        stream_handle=None,
        owner_thread_id=None,
    ):
        self.token_count = token_count
        self.static_bound_values = static_bound_values
        self.static_bound_snapshots = tuple(
            MegaMHCTensorBinding.from_tensor(value) for value in static_bound_values
        )
        self.dynamic_alignments = dynamic_alignments
        self.static_alignments = static_alignments
        self.device = device
        self.stream_handle = stream_handle
        self.owner_thread_id = (
            threading.get_ident() if owner_thread_id is None else owner_thread_id
        )
        self.validation_calls = 0
        self.static_validation_calls = 0
        self.runs = []

    def validate_environment(self):
        self.validation_calls += 1

    def validate_static_bindings(self, values):
        self.static_validation_calls += 1
        if any(
            not snapshot.matches(value)
            for snapshot, value in zip(self.static_bound_snapshots, values, strict=True)
        ):
            raise RuntimeError("static bound value changed after launcher preparation")

    def run_trusted(self, dynamic_values):
        self.runs.append(tuple(dynamic_values))


def _make_case(
    *,
    tokens=1,
    supported_tokens=None,
    profile=DEEPSEEK_V4_FLASH_PROFILE,
    stream_handle=None,
    launcher=None,
    trusted_fast_path=False,
):
    hidden = DEEPSEEK_V4_FLASH_PROFILE.hidden_size
    streams = DEEPSEEK_V4_FLASH_PROFILE.residual_streams
    projection_outputs = DEEPSEEK_V4_FLASH_PROFILE.projection_outputs
    tensors = {
        "x": torch.empty((tokens, hidden), dtype=torch.bfloat16),
        "residual": torch.empty((tokens, streams, hidden), dtype=torch.bfloat16),
        "post_layer_mix": torch.empty((tokens, streams, 1), dtype=torch.float32),
        "comb_res_mix": torch.empty((tokens, streams, streams), dtype=torch.float32),
        "fn": torch.empty((projection_outputs, streams * hidden), dtype=torch.float32),
        "hc_scale": torch.empty((3,), dtype=torch.float32),
        "hc_base": torch.empty((projection_outputs,), dtype=torch.float32),
        "norm_weight": torch.empty((hidden,), dtype=torch.bfloat16),
    }
    launcher = (
        _RecordingLauncher(
            supported_tokens=frozenset(
                {tokens} if supported_tokens is None else supported_tokens
            ),
            stream_handle=stream_handle,
        )
        if launcher is None
        else launcher
    )
    workspace = MegaMHCWorkspace(
        device=torch.device("cpu"),
        stream_handle=stream_handle,
        state={"sentinel": object()},
    )
    context = MegaMHCContext.bind(
        profile=profile,
        fn=tensors["fn"],
        hc_scale=tensors["hc_scale"],
        hc_base=tensors["hc_base"],
        norm_weight=tensors["norm_weight"],
        workspace=workspace,
        launcher=launcher,
        trusted_fast_path=trusted_fast_path,
    )
    return tensors, context, launcher


def _invoke(tensors, context, **overrides):
    arguments = {
        **tensors,
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 20,
        "n_splits": 1,
        "tile_n": 1,
        "norm_eps": 1e-6,
        "context": context,
    }
    arguments.update(overrides)
    return mega_mhc(**arguments)


def _make_outputs(tensors):
    return (
        torch.empty_like(tensors["residual"]),
        torch.empty_like(tensors["post_layer_mix"]),
        torch.empty_like(tensors["comb_res_mix"]),
        torch.empty_like(tensors["x"]),
    )


def _static_values(tensors):
    return (
        tensors["fn"],
        tensors["hc_scale"],
        tensors["hc_base"],
        tensors["norm_weight"],
    )


def test_out_none_allocates_contract_and_enqueues_once():
    tensors, context, launcher = _make_case()

    outputs = _invoke(tensors, context)

    assert len(launcher.calls) == 1
    launch = launcher.calls[0]
    assert launch[0] is context.workspace
    expected_inputs = (
        tensors["x"],
        tensors["residual"],
        tensors["post_layer_mix"],
        tensors["comb_res_mix"],
    )
    assert all(
        actual is expected for actual, expected in zip(launch[1:5], expected_inputs)
    )
    assert all(actual is expected for actual, expected in zip(launch[5:], outputs))
    assert tuple(outputs[0].shape) == tuple(tensors["residual"].shape)
    assert outputs[0].dtype == torch.bfloat16
    assert tuple(outputs[1].shape) == (1, 4, 1)
    assert outputs[1].dtype == torch.float32
    assert tuple(outputs[2].shape) == (1, 4, 4)
    assert outputs[2].dtype == torch.float32
    assert tuple(outputs[3].shape) == tuple(tensors["x"].shape)
    assert outputs[3].dtype == torch.bfloat16


def test_supplied_out_is_returned_and_enqueued_once():
    tensors, context, launcher = _make_case()
    supplied = _make_outputs(tensors)

    outputs = _invoke(tensors, context, out=supplied)

    assert outputs is supplied
    assert len(launcher.calls) == 1
    assert all(
        actual is expected for actual, expected in zip(launcher.calls[0][5:], supplied)
    )


def test_trusted_fast_path_rejects_changed_static_object():
    tensors, context, launcher = _make_case(trusted_fast_path=True)

    with pytest.raises(RuntimeError, match="static tensor object changed"):
        _invoke(tensors, context, fn=tensors["fn"].clone())

    assert launcher.calls == []


def test_trusted_fast_path_rejects_same_object_with_rebound_storage():
    tensors, context, launcher = _make_case(trusted_fast_path=True)
    replacement = torch.empty_like(tensors["fn"])
    tensors["fn"].set_(replacement)

    with pytest.raises(RuntimeError, match="storage/pointer contract changed"):
        _invoke(tensors, context)

    assert launcher.calls == []


def test_trusted_fast_path_rejects_same_object_with_resized_layout():
    tensors, context, launcher = _make_case(trusted_fast_path=True)
    tensors["hc_base"].resize_(12, 2)

    with pytest.raises(RuntimeError, match="storage/pointer contract changed"):
        _invoke(tensors, context)

    assert launcher.calls == []


def test_trusted_fast_path_allows_static_content_update_without_rebinding():
    tensors, context, launcher = _make_case(trusted_fast_path=True)
    original_pointer = tensors["hc_base"].data_ptr()
    tensors["hc_base"].copy_(torch.ones_like(tensors["hc_base"]))

    _invoke(tensors, context)

    assert tensors["hc_base"].data_ptr() == original_pointer
    assert len(launcher.calls) == 1


def test_trusted_fast_path_rejects_dynamic_tensor_abi_mismatch():
    tensors, context, launcher = _make_case(trusted_fast_path=True)
    hidden = DEEPSEEK_V4_FLASH_PROFILE.hidden_size

    with pytest.raises(NotImplementedError, match="x.shape"):
        _invoke(
            tensors,
            context,
            x=torch.empty((1, hidden // 2), dtype=torch.bfloat16),
        )
    with pytest.raises(NotImplementedError, match="x.dtype"):
        _invoke(tensors, context, x=tensors["x"].float())
    with pytest.raises(NotImplementedError, match="residual.dtype"):
        _invoke(tensors, context, residual=tensors["residual"].float())
    with pytest.raises(NotImplementedError, match="post_layer_mix.shape"):
        _invoke(
            tensors,
            context,
            post_layer_mix=torch.empty((1, 4, 2), dtype=torch.float32),
        )
    with pytest.raises(NotImplementedError, match="comb_res_mix.dtype"):
        _invoke(tensors, context, comb_res_mix=tensors["comb_res_mix"].bfloat16())

    assert launcher.calls == []


def test_trusted_fast_path_rejects_dynamic_layout_mismatch():
    tensors, context, launcher = _make_case(tokens=2, trusted_fast_path=True)
    hidden = DEEPSEEK_V4_FLASH_PROFILE.hidden_size
    noncontiguous_x = torch.empty((hidden, 2), dtype=torch.bfloat16).t()
    assert not noncontiguous_x.is_contiguous()

    with pytest.raises(NotImplementedError, match="x must be contiguous"):
        _invoke(tensors, context, x=noncontiguous_x)

    offset_storage = torch.empty((2 * hidden + 8,), dtype=torch.bfloat16)
    offset_x = offset_storage[8:].view(2, hidden)
    assert offset_x.is_contiguous()
    assert offset_x.storage_offset() == 8
    assert offset_x.data_ptr() % 16 == 0
    with pytest.raises(NotImplementedError, match="storage_offset"):
        _invoke(tensors, context, x=offset_x)

    misaligned_buffer = bytearray(2 * hidden * 2 + 2)
    misaligned_x = torch.frombuffer(
        misaligned_buffer,
        dtype=torch.bfloat16,
        count=2 * hidden,
        offset=2,
    ).view(2, hidden)
    assert misaligned_x.storage_offset() == 0
    assert misaligned_x.data_ptr() % 16 != 0
    with pytest.raises(NotImplementedError, match="aligned"):
        _invoke(tensors, context, x=misaligned_x)

    assert launcher.calls == []


def test_context_rejects_calls_from_a_different_host_thread():
    tensors, context, launcher = _make_case(trusted_fast_path=True)
    errors = []

    def invoke_from_other_thread():
        try:
            _invoke(tensors, context)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    thread = threading.Thread(target=invoke_from_other_thread)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "different host thread" in str(errors[0])
    assert launcher.calls == []


def test_trusted_fast_path_still_validates_supplied_outputs():
    tensors, context, launcher = _make_case(trusted_fast_path=True)
    outputs = list(_make_outputs(tensors))
    outputs[3] = torch.empty(tensors["x"].shape, dtype=torch.float32)

    with pytest.raises(ValueError, match="y_bf16.dtype"):
        _invoke(tensors, context, out=tuple(outputs))

    assert launcher.calls == []


def test_trusted_fast_path_rejects_offset_supplied_output():
    tensors, context, launcher = _make_case(trusted_fast_path=True)
    outputs = list(_make_outputs(tensors))
    hidden = DEEPSEEK_V4_FLASH_PROFILE.hidden_size
    offset_storage = torch.empty((hidden + 8,), dtype=torch.bfloat16)
    outputs[3] = offset_storage[8:].view(1, hidden)
    assert outputs[3].is_contiguous()
    assert outputs[3].storage_offset() == 8

    with pytest.raises(ValueError, match="storage_offset"):
        _invoke(tensors, context, out=tuple(outputs))

    misaligned_buffer = bytearray(hidden * 2 + 2)
    outputs[3] = torch.frombuffer(
        misaligned_buffer,
        dtype=torch.bfloat16,
        count=hidden,
        offset=2,
    ).view(1, hidden)
    assert outputs[3].storage_offset() == 0
    assert outputs[3].data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="aligned"):
        _invoke(tensors, context, out=tuple(outputs))

    assert launcher.calls == []


def test_trusted_context_validates_static_contract_at_bind():
    tensors, _, launcher = _make_case()
    workspace = MegaMHCWorkspace(
        device=torch.device("cpu"), stream_handle=None, state={}
    )

    with pytest.raises(NotImplementedError, match="norm_weight.dtype"):
        MegaMHCContext.bind(
            profile=DEEPSEEK_V4_FLASH_PROFILE,
            fn=tensors["fn"],
            hc_scale=tensors["hc_scale"],
            hc_base=tensors["hc_base"],
            norm_weight=tensors["norm_weight"].float(),
            workspace=workspace,
            launcher=launcher,
            trusted_fast_path=True,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_splits": 2}, "n_splits"),
        ({"n_splits": 1.5}, "n_splits"),
        ({"tile_n": 2}, "tile_n"),
        ({"tile_n": 1.5}, "tile_n"),
        ({"sinkhorn_repeat": 19}, "sinkhorn_repeat"),
        ({"sinkhorn_repeat": 20.5}, "sinkhorn_repeat"),
        ({"hc_pre_eps": 1e-5}, "hc_pre_eps"),
    ],
)
def test_unsupported_numerical_contract_is_explicit(override, message):
    tensors, context, launcher = _make_case()

    with pytest.raises(NotImplementedError, match=message):
        _invoke(tensors, context, **override)

    assert launcher.calls == []


def test_unsupported_token_specialization_is_explicit():
    tensors, context, launcher = _make_case(tokens=2, supported_tokens=frozenset({1}))

    with pytest.raises(NotImplementedError, match="T=2"):
        _invoke(tensors, context)

    assert launcher.calls == []


def test_unsupported_profile_is_explicit():
    other_profile = replace(
        DEEPSEEK_V4_FLASH_PROFILE, model_revision="different-revision"
    )
    with pytest.raises(NotImplementedError, match="DeepSeek-V4-Flash"):
        _make_case(profile=other_profile)


def test_autograd_inputs_are_not_silently_detached():
    tensors, context, launcher = _make_case()
    grad_x = tensors["x"].clone().requires_grad_(True)

    with pytest.raises(NotImplementedError, match="must not require gradients"):
        _invoke(tensors, context, x=grad_x)

    assert launcher.calls == []


def test_poisoned_workspace_is_never_reused():
    tensors, context, launcher = _make_case()
    context.workspace.poisoned = True

    with pytest.raises(RuntimeError, match="workspace is poisoned"):
        _invoke(tensors, context)

    assert launcher.calls == []


def test_non_cuda_workspace_rejects_cuda_stream_handle():
    with pytest.raises(ValueError, match="non-CUDA"):
        _make_case(stream_handle=1234)


def test_supplied_out_must_be_a_four_tensor_tuple():
    tensors, context, launcher = _make_case()

    with pytest.raises(TypeError, match="tuple"):
        _invoke(tensors, context, out=list(_make_outputs(tensors)))
    with pytest.raises(ValueError, match="exactly four"):
        _invoke(tensors, context, out=_make_outputs(tensors)[:3])

    assert launcher.calls == []


def test_supplied_outputs_must_not_alias_inputs():
    tensors, context, launcher = _make_case()
    outputs = list(_make_outputs(tensors))
    outputs[3] = tensors["x"]

    with pytest.raises(ValueError, match="must not alias"):
        _invoke(tensors, context, out=tuple(outputs))

    assert launcher.calls == []


def test_supplied_outputs_must_not_share_storage():
    tensors, context, launcher = _make_case()
    outputs = list(_make_outputs(tensors))
    shared = torch.empty((16,), dtype=torch.float32)
    outputs[1] = shared[:4].view(1, 4, 1)
    outputs[2] = shared.view(1, 4, 4)

    with pytest.raises(ValueError, match="must not share storage"):
        _invoke(tensors, context, out=tuple(outputs))

    assert launcher.calls == []


def test_synchronous_launch_failure_poisons_workspace():
    failing_launcher = _RaisingLauncher()
    tensors, context, _ = _make_case(launcher=failing_launcher)

    with pytest.raises(RuntimeError, match="synthetic launch failure"):
        _invoke(tensors, context)

    assert len(failing_launcher.calls) == 1
    assert context.workspace.poisoned
    with pytest.raises(RuntimeError, match="workspace is poisoned"):
        _invoke(tensors, context)
    assert len(failing_launcher.calls) == 1


def test_workspace_rejects_different_current_cuda_device(monkeypatch):
    workspace = MegaMHCWorkspace(
        device=torch.device("cuda", 2), stream_handle=0xABC, state={}
    )
    context = SimpleNamespace(
        workspace=workspace,
        owner_thread_id=threading.get_ident(),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    def unexpected_stream_query(_device):
        raise AssertionError("stream must not be queried on the wrong current device")

    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_stream_query)
    with pytest.raises(RuntimeError, match="different CUDA device"):
        megamhc_api._validate_workspace(context)


def test_context_requires_at_least_one_compiled_token_specialization():
    tensors, _, _ = _make_case()
    launcher = _RecordingLauncher(supported_tokens=frozenset())
    workspace = MegaMHCWorkspace(
        device=torch.device("cpu"), stream_handle=None, state={}
    )

    with pytest.raises(ValueError, match="supported_tokens"):
        MegaMHCContext.bind(
            profile=DEEPSEEK_V4_FLASH_PROFILE,
            fn=tensors["fn"],
            hc_scale=tensors["hc_scale"],
            hc_base=tensors["hc_base"],
            norm_weight=tensors["norm_weight"],
            workspace=workspace,
            launcher=launcher,
        )


def test_context_rejects_nonpositive_token_specializations():
    tensors, _, _ = _make_case()
    launcher = _RecordingLauncher(supported_tokens=frozenset({0, 1}))
    workspace = MegaMHCWorkspace(
        device=torch.device("cpu"), stream_handle=None, state={}
    )

    with pytest.raises(ValueError, match="positive"):
        MegaMHCContext.bind(
            profile=DEEPSEEK_V4_FLASH_PROFILE,
            fn=tensors["fn"],
            hc_scale=tensors["hc_scale"],
            hc_base=tensors["hc_base"],
            norm_weight=tensors["norm_weight"],
            workspace=workspace,
            launcher=launcher,
        )


def test_context_rejects_fractional_token_specializations():
    tensors, _, _ = _make_case()
    launcher = _RecordingLauncher(supported_tokens=frozenset({1.5}))
    workspace = MegaMHCWorkspace(
        device=torch.device("cpu"), stream_handle=None, state={}
    )

    with pytest.raises(TypeError, match="exact integers"):
        MegaMHCContext.bind(
            profile=DEEPSEEK_V4_FLASH_PROFILE,
            fn=tensors["fn"],
            hc_scale=tensors["hc_scale"],
            hc_base=tensors["hc_base"],
            norm_weight=tensors["norm_weight"],
            workspace=workspace,
            launcher=launcher,
        )


def test_factory_derives_token_and_dispatches_exact_prepared_launcher():
    tensors, _, _ = _make_case()
    static_values = _static_values(tensors)
    prepared = _PreparedLauncher(1, static_values)
    context = _create_context_from_prepared(*static_values, prepared)

    assert context.supported_tokens == frozenset({1})
    assert context.launcher.dynamic_alignments(1) == (16,) * 8
    assert context.launcher.static_alignments == (16,) * 4
    assert prepared.validation_calls == 1
    assert prepared.static_validation_calls == 1
    _invoke(tensors, context)
    assert len(prepared.runs) == 1


def test_factory_rejects_launcher_bound_to_different_static_tensor():
    tensors, _, _ = _make_case()
    static_values = _static_values(tensors)
    rebound_values = (tensors["fn"].clone(), *static_values[1:])
    prepared = _PreparedLauncher(1, rebound_values)

    with pytest.raises(RuntimeError, match="different static tensors"):
        _create_context_from_prepared(*static_values, prepared)


def test_factory_rejects_static_storage_rebind_after_launcher_preparation():
    tensors, _, _ = _make_case()
    static_values = _static_values(tensors)
    prepared = _PreparedLauncher(1, static_values)
    replacement = tensors["fn"].clone()
    tensors["fn"].set_(replacement)

    with pytest.raises(RuntimeError, match="changed after launcher preparation"):
        _create_context_from_prepared(*static_values, prepared)


def test_factory_rejects_wrong_owner_and_invalid_alignment_contracts():
    tensors, _, _ = _make_case()
    static_values = _static_values(tensors)
    wrong_owner = _PreparedLauncher(
        1, static_values, owner_thread_id=threading.get_ident() + 1
    )
    with pytest.raises(RuntimeError, match="different host thread"):
        _create_context_from_prepared(*static_values, wrong_owner)

    invalid_alignment = _PreparedLauncher(
        1, static_values, dynamic_alignments=(16,) * 7 + (3,)
    )
    with pytest.raises(ValueError, match="powers of two"):
        _create_context_from_prepared(*static_values, invalid_alignment)


def test_factory_missing_token_is_explicit():
    tensors, _, _ = _make_case(tokens=2)
    static_values = _static_values(tensors)
    prepared = _PreparedLauncher(1, static_values)
    context = _create_context_from_prepared(*static_values, prepared)
    with pytest.raises(NotImplementedError, match="T=2"):
        _invoke(tensors, context)
    assert prepared.runs == []


def _compiled_for_admission(
    *,
    num_warps=20,
    shared=0,
    cluster_dims=(1, 1, 1),
    launch_cooperative_grid=False,
    num_ctas=1,
):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            num_warps=num_warps,
            shared=shared,
            cluster_dims=cluster_dims,
            launch_cooperative_grid=launch_cooperative_grid,
            num_ctas=num_ctas,
        )
    )


def _spec_for_admission(
    *,
    cooperative=False,
    cluster_dims=None,
):
    kwargs = {"num_ctas": 1}
    if cooperative:
        kwargs["launch_cooperative_grid"] = True
    if cluster_dims is not None:
        kwargs["cluster_dims"] = cluster_dims
    return SimpleNamespace(
        kwargs=kwargs,
        label="host-admission-test",
        requires_full_grid_residency=cooperative,
    )


def test_cooperative_full_grid_admission_accepts_exact_resident_capacity():
    occupancy_queries = []
    compiled = _compiled_for_admission(
        num_warps=20, shared=24576, launch_cooperative_grid=True
    )
    runtime = SimpleNamespace(
        cooperative_launch_supported=lambda device: device == 2,
        device_properties=lambda device: {"multiprocessor_count": 132},
        active_blocks_per_sm=lambda kernel, threads, shared: (
            occupancy_queries.append((kernel, threads, shared)) or 2
        ),
    )

    admission = prepared_launcher_module._admit_cooperative_grid(
        _spec_for_admission(cooperative=True),
        compiled,
        (264, 1, 1),
        runtime,
        2,
    )

    assert admission == prepared_launcher_module.CooperativeGridAdmission(
        grid_ctas=264,
        active_blocks_per_sm=2,
        multiprocessor_count=132,
        resident_cta_capacity=264,
    )
    assert occupancy_queries == [(compiled, 640, 24576)]


def test_cooperative_full_grid_admission_rejects_oversubscribed_grid():
    compiled = _compiled_for_admission(launch_cooperative_grid=True)
    runtime = SimpleNamespace(
        cooperative_launch_supported=lambda device: True,
        device_properties=lambda device: {"multiprocessor_count": 132},
        active_blocks_per_sm=lambda kernel, threads, shared: 2,
    )

    with pytest.raises(
        prepared_launcher_module.UnsupportedPreparedRuntimeError,
        match=r"265 CTAs.*264 resident CTAs",
    ):
        prepared_launcher_module._admit_cooperative_grid(
            _spec_for_admission(cooperative=True),
            compiled,
            (265, 1, 1),
            runtime,
            2,
        )


@pytest.mark.parametrize(("active_blocks", "resident_capacity"), [(2, 228), (0, 0)])
def test_factory_reports_insufficient_cooperative_residency_as_unsupported(
    monkeypatch, active_blocks, resident_capacity
):
    from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamhc.tle import (
        isolated_dispatch,
    )

    def prepare(_inputs, _outputs):
        runtime = SimpleNamespace(
            cooperative_launch_supported=lambda device: True,
            device_properties=lambda device: {"multiprocessor_count": 114},
            active_blocks_per_sm=lambda kernel, threads, shared: active_blocks,
        )
        return prepared_launcher_module._admit_cooperative_grid(
            _spec_for_admission(cooperative=True),
            _compiled_for_admission(num_warps=16, launch_cooperative_grid=True),
            (264, 1, 1),
            runtime,
            0,
        )

    cuda = SimpleNamespace(
        device=lambda device: nullcontext(),
        is_current_stream_capturing=lambda: False,
        get_device_capability=lambda device: (9, 0),
    )
    tensor = SimpleNamespace(device=torch.device("cuda:0"))
    monkeypatch.setattr(
        megamhc_factory,
        "torch",
        SimpleNamespace(Tensor=SimpleNamespace, cuda=cuda, empty_like=lambda t: t),
    )
    monkeypatch.setattr(isolated_dispatch, "prepare_isolated_safe_launcher", prepare)

    with pytest.raises(
        NotImplementedError,
        match=rf"264 CTAs.*114 SMs \({resident_capacity} resident CTAs\)",
    ) as failure:
        megamhc_factory.create_mega_mhc_context(*(tensor,) * 8)
    assert isinstance(
        failure.value.__cause__,
        prepared_launcher_module.UnsupportedPreparedRuntimeError,
    )


def test_cooperative_full_grid_admission_rejects_unsupported_device_first():
    def unexpected_query(*args, **kwargs):
        raise AssertionError("occupancy must not be queried on unsupported devices")

    runtime = SimpleNamespace(
        cooperative_launch_supported=lambda device: False,
        device_properties=unexpected_query,
        active_blocks_per_sm=unexpected_query,
    )

    with pytest.raises(
        prepared_launcher_module.UnsupportedPreparedRuntimeError,
        match="does not support it",
    ):
        prepared_launcher_module._admit_cooperative_grid(
            _spec_for_admission(cooperative=True),
            _compiled_for_admission(launch_cooperative_grid=True),
            (1, 1, 1),
            runtime,
            2,
        )


def test_cooperative_full_grid_admission_rejects_noncooperative_compilation():
    with pytest.raises(
        prepared_launcher_module.UnsupportedPreparedRuntimeError,
        match="compiled kernel does not enable it",
    ):
        prepared_launcher_module._admit_cooperative_grid(
            _spec_for_admission(cooperative=True),
            _compiled_for_admission(),
            (264, 1, 1),
            SimpleNamespace(),
            2,
        )


def test_cooperative_full_grid_admission_rejects_multiple_compiled_ctas_per_program():
    runtime = SimpleNamespace(cooperative_launch_supported=lambda device: True)
    with pytest.raises(
        prepared_launcher_module.UnsupportedPreparedRuntimeError,
        match="one CTA per program",
    ):
        prepared_launcher_module._admit_cooperative_grid(
            _spec_for_admission(cooperative=True),
            _compiled_for_admission(launch_cooperative_grid=True, num_ctas=2),
            (264, 1, 1),
            runtime,
            2,
        )


@pytest.mark.parametrize(("num_warps", "shared"), [(0, 0), (20, 1.5)])
def test_compiled_launch_resources_reject_invalid_metadata(num_warps, shared):
    with pytest.raises(
        prepared_launcher_module.UnsupportedPreparedRuntimeError,
        match="num_warps/shared occupancy metadata",
    ):
        prepared_launcher_module._compiled_launch_resources(
            _compiled_for_admission(num_warps=num_warps, shared=shared)
        )


@pytest.mark.parametrize(
    ("cluster_size", "maximum_cluster_size"),
    [(16, 16), (8, 8)],
)
def test_cluster_admission_accepts_supported_cluster_sizes(
    cluster_size,
    maximum_cluster_size,
):
    cluster_dims = (cluster_size, 1, 1)
    compiled = _compiled_for_admission(
        num_warps=16,
        shared=32768,
        cluster_dims=cluster_dims,
    )
    queries = []
    runtime = SimpleNamespace(
        maximum_cluster_size=lambda kernel, threads, shared, grid, dims: (
            queries.append((kernel, threads, shared, grid, dims))
            or maximum_cluster_size
        )
    )

    admission = prepared_launcher_module._admit_cluster_launch(
        _spec_for_admission(cluster_dims=cluster_dims),
        compiled,
        (128, 1, 1),
        runtime,
    )

    assert admission == prepared_launcher_module.ClusterLaunchAdmission(
        cluster_dims=cluster_dims,
        required_cluster_size=cluster_size,
        maximum_cluster_size=maximum_cluster_size,
    )
    assert queries == [(compiled, 512, 32768, (128, 1, 1), cluster_dims)]


@pytest.mark.parametrize("maximum_cluster_size", [8, 0])
def test_cluster_admission_rejects_cluster_larger_than_device_limit(
    maximum_cluster_size,
):
    cluster_dims = (16, 1, 1)
    runtime = SimpleNamespace(
        maximum_cluster_size=lambda kernel, threads, shared, grid, dims: (
            maximum_cluster_size
        )
    )

    with pytest.raises(
        prepared_launcher_module.UnsupportedPreparedRuntimeError,
        match=rf"requires a 16-CTA cluster.*at most {maximum_cluster_size}",
    ):
        prepared_launcher_module._admit_cluster_launch(
            _spec_for_admission(cluster_dims=cluster_dims),
            _compiled_for_admission(num_warps=16, cluster_dims=cluster_dims),
            (128, 1, 1),
            runtime,
        )


def test_occupancy_query_preserves_zero_and_reports_resource_exhaustion(monkeypatch):
    from triton.runtime.errors import OutOfResources

    original_import = builtins.__import__
    driver = SimpleNamespace(
        cuOccupancyMaxActiveBlocksPerMultiprocessor=lambda *args: (0, 0)
    )

    def import_with_cuda_stub(name, *args, **kwargs):
        if name == "cuda.bindings":
            return SimpleNamespace(driver=driver)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_cuda_stub)
    compiled = SimpleNamespace(function=object(), _init_handles=lambda: None)
    assert prepared_launcher_module._active_blocks_per_sm(compiled, 640, 24576) == 0

    def out_of_resources():
        raise OutOfResources(24576, 16384, "shared memory")

    compiled._init_handles = out_of_resources
    with pytest.raises(
        prepared_launcher_module.UnsupportedPreparedRuntimeError,
        match="shared memory",
    ) as failure:
        prepared_launcher_module._active_blocks_per_sm(compiled, 640, 24576)
    assert isinstance(failure.value.__cause__, OutOfResources)


def test_cluster_capability_query_fails_closed_without_cuda_bindings(monkeypatch):
    original_import = builtins.__import__

    def reject_cuda_bindings(name, *args, **kwargs):
        if name == "cuda.bindings":
            raise ImportError("synthetic missing cuda.bindings")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_cuda_bindings)
    with pytest.raises(
        prepared_launcher_module.UnsupportedPreparedRuntimeError,
        match="cluster capability cannot be queried",
    ):
        prepared_launcher_module._maximum_cluster_size(
            _compiled_for_admission(num_warps=16, cluster_dims=(16, 1, 1)),
            512,
            0,
            (128, 1, 1),
            (16, 1, 1),
        )
