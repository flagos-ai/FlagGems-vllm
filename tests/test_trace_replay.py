# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

"""Accuracy tests for trace replay."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import apply_trace_tokens as flaggems_apply_trace_tokens

try:
    from vllm.v1.worker.gpu.sample.trace_replay import (
        apply_trace_tokens as vllm_reference,
    )
except (ImportError, OSError):
    vllm_reference = None


DEVICE = flaggems_vllm.device
_device_module = getattr(torch, DEVICE, None)
_HAS_DEVICE = _device_module is not None and _device_module.is_available()
pytestmark = [
    pytest.mark.skipif(
        not _HAS_DEVICE, reason=f"requires an available {DEVICE} device"
    ),
]

MAX_MODEL_LEN = 2048


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device=DEVICE)


def _i64(values):
    return torch.tensor(values, dtype=torch.int64, device=DEVICE)


def _cpu_reference(sampled, mapping, traces, lengths, total, prompt):
    """Compute expected sampled tokens on CPU."""
    result = sampled.cpu().tolist()
    request_indices = mapping.cpu().tolist()
    trace_rows = traces.cpu().tolist()
    trace_lengths = lengths.cpu().tolist()
    total_lengths = total.cpu().tolist()
    prompt_lengths = prompt.cpu().tolist()
    for batch_idx, req_idx in enumerate(request_indices):
        if req_idx < 0:
            continue
        step = total_lengths[req_idx] - prompt_lengths[req_idx]
        if 0 <= step < trace_lengths[req_idx]:
            result[batch_idx] = trace_rows[req_idx][step]
    return result


def _assert_replay(sampled, mapping, traces, lengths, total, prompt, expected=None):
    cpu_expected = _cpu_reference(sampled, mapping, traces, lengths, total, prompt)
    if expected is not None:
        assert cpu_expected == expected
    reference_sampled = sampled.clone() if vllm_reference is not None else None
    inputs = {
        "mapping": mapping,
        "traces": traces,
        "lengths": lengths,
        "total": total,
        "prompt": prompt,
    }
    inputs_before = {name: tensor.clone() for name, tensor in inputs.items()}
    pointer = sampled.data_ptr()
    assert (
        flaggems_apply_trace_tokens(sampled, mapping, traces, lengths, total, prompt)
        is None
    )
    assert sampled.data_ptr() == pointer
    assert sampled.cpu().tolist() == cpu_expected
    for name, before in inputs_before.items():
        assert torch.equal(
            inputs[name], before
        ), f"apply_trace_tokens must not modify {name}"

    if reference_sampled is not None:
        assert (
            vllm_reference(reference_sampled, mapping, traces, lengths, total, prompt)
            is None
        )
        assert reference_sampled.cpu().tolist() == cpu_expected


@pytest.mark.apply_trace_tokens
def test_replay_overwrites_sampled_at_each_step():
    traces = torch.zeros(2, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    traces[0, :3] = _i32([100, 101, 102])
    traces[1, :3] = _i32([200, 201, 202])
    lengths = _i32([3, 3])
    prompt = _i32([5, 8])
    mapping = _i32([0, 1])

    for step in range(3):
        _assert_replay(
            _i64([-7, -7]),
            mapping,
            traces,
            lengths,
            _i32([5 + step, 8 + step]),
            prompt,
            [100 + step, 200 + step],
        )


@pytest.mark.apply_trace_tokens
def test_past_end_of_trace_leaves_sampled_untouched():
    traces = torch.zeros(1, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    traces[0, :2] = _i32([100, 101])
    _assert_replay(
        _i64([999]), _i32([0]), traces, _i32([2]), _i32([6]), _i32([4]), [999]
    )


@pytest.mark.apply_trace_tokens
def test_non_trace_request_untouched():
    traces = torch.zeros(1, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    _assert_replay(_i64([42]), _i32([0]), traces, _i32([0]), _i32([3]), _i32([3]), [42])


@pytest.mark.apply_trace_tokens
def test_idx_mapping_indirection_and_negative_skip():
    traces = torch.zeros(2, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    traces[1, :2] = _i32([500, 501])
    _assert_replay(
        _i64([1, 2, 3]),
        _i32([1, -1, 0]),
        traces,
        _i32([0, 2]),
        _i32([10, 7]),
        _i32([10, 6]),
        [501, 2, 3],
    )


@pytest.mark.apply_trace_tokens
@pytest.mark.parametrize("num_reqs", [5, 33, 127, 128, 129, 256, 257])
def test_mixed_requests_with_padded_trace_rows(num_reqs):
    row_width = MAX_MODEL_LEN + 3
    traces = torch.arange(
        (num_reqs + 1) * row_width, dtype=torch.int32, device=DEVICE
    ).reshape(num_reqs + 1, row_width)[:, 1:-2]
    mapping = _i32(list(reversed(range(num_reqs))))
    mapping[::11] = -1
    lengths = _i32([idx % 4 for idx in range(num_reqs)])
    prompt = _i32([5] * num_reqs)
    total = _i32([4 + idx % 6 for idx in range(num_reqs)])
    _assert_replay(_i64([-7] * num_reqs), mapping, traces, lengths, total, prompt)


@pytest.mark.apply_trace_tokens
def test_repeated_request_mapping_and_negative_token():
    traces = torch.zeros(2, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    traces[0, 0] = -1
    traces[1, 1] = -123456789
    _assert_replay(
        _i64([11, 12, 13, 14, 15]),
        _i32([1, 1, -1, 0, 1]),
        traces,
        _i32([1, 2]),
        _i32([5, 6]),
        _i32([5, 5]),
        [-123456789, -123456789, 13, -1, -123456789],
    )


@pytest.mark.apply_trace_tokens
def test_inactive_lanes_preserve_out_of_int32_range_int64_sampled():
    """Inactive lanes keep the original int64 value, including values > 2**31."""
    traces = torch.zeros(2, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    traces[0, :2] = _i32([100, 101])
    big = 1 << 40
    # lane 0: replayed -> 101
    # lane 1: negative mapping, must keep -big
    # lane 2: trace_len == 0, must keep big + 3
    _assert_replay(
        _i64([big, -big, big + 3]),
        _i32([0, -1, 1]),
        traces,
        _i32([2, 0]),
        _i32([3, 4]),
        _i32([2, 4]),
        [101, -big, big + 3],
    )


@pytest.mark.apply_trace_tokens
def test_export_and_public_entry():
    assert "apply_trace_tokens" in flaggems_vllm.ops.__all__
    assert callable(flaggems_vllm.ops.apply_trace_tokens)
    assert callable(flaggems_vllm.apply_trace_tokens)


@pytest.mark.apply_trace_tokens
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "enflame",
    reason="Enflame specialization is only registered on Enflame GCU",
)
def test_enflame_vendor_dispatch():
    from flaggems_vllm.runtime.backend._enflame.ops.trace_replay import (
        apply_trace_tokens as enflame_apply_trace_tokens,
    )

    assert flaggems_vllm.apply_trace_tokens is enflame_apply_trace_tokens
    assert flaggems_vllm.apply_trace_tokens.__module__ == (
        "flaggems_vllm.runtime.backend._enflame.ops.trace_replay"
    )
