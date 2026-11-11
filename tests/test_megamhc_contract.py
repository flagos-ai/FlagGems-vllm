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

"""CPU tests for the unexported Hopper MegaMHC host contract."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest
import torch

from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamhc import (
    DEEPSEEK_V4_FLASH_PROFILE,
    MegaMHCContext,
    MegaMHCWorkspace,
    mega_mhc,
)


class _RecordingLauncher:
    def __init__(self):
        self.calls = []

    def launch(self, *args):
        self.calls.append(args)


class _RaisingLauncher(_RecordingLauncher):
    def launch(self, *args):
        super().launch(*args)
        raise RuntimeError("synthetic launch failure")


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
    launcher = _RecordingLauncher() if launcher is None else launcher
    workspace = MegaMHCWorkspace(
        device=torch.device("cpu"),
        stream_handle=stream_handle,
        state={"sentinel": object()},
    )
    context = MegaMHCContext.bind(
        profile=profile,
        supported_tokens=frozenset(
            {tokens} if supported_tokens is None else supported_tokens
        ),
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


def test_trusted_fast_path_allocates_and_enqueues_once(monkeypatch):
    tensors, context, launcher = _make_case(trusted_fast_path=True)
    api_module = __import__(mega_mhc.__module__, fromlist=["api"])

    def unexpected_strict_validation(*args, **kwargs):
        del args, kwargs
        raise AssertionError("trusted path performed strict per-call validation")

    monkeypatch.setattr(api_module, "_validate_profile", unexpected_strict_validation)
    monkeypatch.setattr(api_module, "_validate_outputs", unexpected_strict_validation)

    outputs = _invoke(tensors, context)

    assert len(launcher.calls) == 1
    assert all(
        actual is expected for actual, expected in zip(launcher.calls[0][5:], outputs)
    )


def test_trusted_fast_path_rejects_changed_static_object():
    tensors, context, launcher = _make_case(trusted_fast_path=True)

    with pytest.raises(RuntimeError, match="static tensor object changed"):
        _invoke(tensors, context, fn=tensors["fn"].clone())

    assert launcher.calls == []


def test_trusted_fast_path_rejects_unsupported_token_and_scalar():
    tensors, context, launcher = _make_case(
        tokens=2,
        supported_tokens=frozenset({1}),
        trusted_fast_path=True,
    )

    with pytest.raises(NotImplementedError, match="T=2"):
        _invoke(tensors, context)
    with pytest.raises(NotImplementedError, match="n_splits"):
        _invoke(
            tensors,
            context,
            x=tensors["x"][:1],
            residual=tensors["residual"][:1],
            post_layer_mix=tensors["post_layer_mix"][:1],
            comb_res_mix=tensors["comb_res_mix"][:1],
            n_splits=1.5,
        )

    assert launcher.calls == []


def test_trusted_fast_path_rejects_dynamic_tensor_abi_mismatch():
    tensors, context, launcher = _make_case(trusted_fast_path=True)
    hidden = DEEPSEEK_V4_FLASH_PROFILE.hidden_size

    with pytest.raises(NotImplementedError, match="x.shape"):
        _invoke(
            tensors,
            context,
            x=torch.empty((1, hidden // 2), dtype=torch.bfloat16),
        )
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
            supported_tokens=frozenset({1}),
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
    tensors, context, launcher = _make_case(profile=other_profile)

    with pytest.raises(NotImplementedError, match="DeepSeek-V4-Flash"):
        _invoke(tensors, context)

    assert launcher.calls == []


def test_unsupported_dtype_and_layout_are_explicit():
    tensors, context, launcher = _make_case(tokens=2)

    with pytest.raises(NotImplementedError, match="x.dtype"):
        _invoke(tensors, context, x=tensors["x"].float())
    noncontiguous_x = torch.empty(
        (DEEPSEEK_V4_FLASH_PROFILE.hidden_size, 2), dtype=torch.bfloat16
    ).t()
    assert not noncontiguous_x.is_contiguous()
    with pytest.raises(NotImplementedError, match="x must be contiguous"):
        _invoke(tensors, context, x=noncontiguous_x)

    assert launcher.calls == []


def test_unsupported_hidden_shape_is_explicit():
    tensors, context, launcher = _make_case()
    wrong_hidden = torch.empty(
        (1, DEEPSEEK_V4_FLASH_PROFILE.hidden_size // 2), dtype=torch.bfloat16
    )

    with pytest.raises(NotImplementedError, match="x.shape"):
        _invoke(tensors, context, x=wrong_hidden)

    assert launcher.calls == []


def test_autograd_inputs_are_not_silently_detached():
    tensors, context, launcher = _make_case()
    grad_x = tensors["x"].clone().requires_grad_(True)

    with pytest.raises(NotImplementedError, match="must not require gradients"):
        _invoke(tensors, context, x=grad_x)

    assert launcher.calls == []


def test_static_tensor_rebinding_requires_a_new_context():
    tensors, context, launcher = _make_case()

    with pytest.raises(RuntimeError, match="static tensor identity changed"):
        _invoke(tensors, context, fn=tensors["fn"].clone())

    assert launcher.calls == []


def test_poisoned_workspace_is_never_reused():
    tensors, context, launcher = _make_case()
    context.workspace.poisoned = True

    with pytest.raises(RuntimeError, match="workspace is poisoned"):
        _invoke(tensors, context)

    assert launcher.calls == []


def test_non_cuda_workspace_rejects_cuda_stream_handle():
    tensors, context, launcher = _make_case(stream_handle=1234)

    with pytest.raises(ValueError, match="non-CUDA"):
        _invoke(tensors, context)

    assert launcher.calls == []


def test_invalid_supplied_outputs_do_not_launch():
    tensors, context, launcher = _make_case()
    outputs = list(_make_outputs(tensors))
    outputs[3] = torch.empty(tensors["x"].shape, dtype=torch.float32)

    with pytest.raises(ValueError, match="y_bf16.dtype"):
        _invoke(tensors, context, out=tuple(outputs))

    assert launcher.calls == []


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


def test_context_requires_at_least_one_compiled_token_specialization():
    tensors, _, launcher = _make_case()
    workspace = MegaMHCWorkspace(
        device=torch.device("cpu"), stream_handle=None, state={}
    )

    with pytest.raises(ValueError, match="supported_tokens"):
        MegaMHCContext.bind(
            profile=DEEPSEEK_V4_FLASH_PROFILE,
            supported_tokens=frozenset(),
            fn=tensors["fn"],
            hc_scale=tensors["hc_scale"],
            hc_base=tensors["hc_base"],
            norm_weight=tensors["norm_weight"],
            workspace=workspace,
            launcher=launcher,
        )


def test_context_rejects_nonpositive_token_specializations():
    tensors, _, launcher = _make_case()
    workspace = MegaMHCWorkspace(
        device=torch.device("cpu"), stream_handle=None, state={}
    )

    with pytest.raises(ValueError, match="positive"):
        MegaMHCContext.bind(
            profile=DEEPSEEK_V4_FLASH_PROFILE,
            supported_tokens=frozenset({0, 1}),
            fn=tensors["fn"],
            hc_scale=tensors["hc_scale"],
            hc_base=tensors["hc_base"],
            norm_weight=tensors["norm_weight"],
            workspace=workspace,
            launcher=launcher,
        )


def test_context_rejects_fractional_token_specializations():
    tensors, _, launcher = _make_case()
    workspace = MegaMHCWorkspace(
        device=torch.device("cpu"), stream_handle=None, state={}
    )

    with pytest.raises(TypeError, match="exact integers"):
        MegaMHCContext.bind(
            profile=DEEPSEEK_V4_FLASH_PROFILE,
            supported_tokens=frozenset({1.5}),
            fn=tensors["fn"],
            hc_scale=tensors["hc_scale"],
            hc_base=tensors["hc_base"],
            norm_weight=tensors["norm_weight"],
            workspace=workspace,
            launcher=launcher,
        )
