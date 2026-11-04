# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy tests for trace replay."""

import pytest
import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm import apply_trace_tokens as flaggems_apply_trace_tokens


@triton.jit
def _trace_replay_kernel(
    sampled_ptr,  # [num_reqs], int64, mutated in place
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx
    trace_token_ids_ptr,  # [max_num_reqs, max_model_len], int32
    trace_token_ids_stride,
    trace_len_ptr,  # [max_num_reqs], int32
    total_len_ptr,  # [max_num_reqs], int32
    prompt_len_ptr,  # [max_num_reqs], int32
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return

    trace_len = tl.load(trace_len_ptr + req_state_idx)
    if trace_len <= 0:
        return

    # Subtract prompt tokens to find the current decode step.
    step = tl.load(total_len_ptr + req_state_idx) - tl.load(
        prompt_len_ptr + req_state_idx
    )
    if step < 0 or step >= trace_len:
        return

    token_id = tl.load(
        trace_token_ids_ptr + req_state_idx * trace_token_ids_stride + step
    )
    tl.store(sampled_ptr + batch_idx, token_id.to(tl.int64))


def apply_trace_tokens(
    sampled: torch.Tensor,
    idx_mapping: torch.Tensor,
    trace_token_ids: torch.Tensor,
    trace_len: torch.Tensor,
    total_len: torch.Tensor,
    prompt_len: torch.Tensor,
) -> None:
    """Overwrite ``sampled`` in place with trace tokens for the current step."""
    num_reqs = idx_mapping.shape[0]
    _trace_replay_kernel[(num_reqs,)](
        sampled,
        idx_mapping,
        trace_token_ids,
        trace_token_ids.stride(0),
        trace_len,
        total_len,
        prompt_len,
    )


_kernel_mark = pytest.MarkDecorator(
    pytest.Mark("_trace_replay_kernel", (), {}, _ispytest=True),
    _ispytest=True,
)
DEVICE = flaggems_vllm.device
_device_module = getattr(torch, DEVICE, None)
_HAS_DEVICE = _device_module is not None and _device_module.is_available()
pytestmark = [
    pytest.mark.apply_trace_tokens,
    _kernel_mark,
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
    reference_sampled = (
        sampled.clone() if flaggems_vllm.vendor_name == "nvidia" else None
    )
    pointer = sampled.data_ptr()
    assert (
        flaggems_apply_trace_tokens(sampled, mapping, traces, lengths, total, prompt)
        is None
    )
    assert sampled.data_ptr() == pointer
    assert sampled.cpu().tolist() == cpu_expected

    if reference_sampled is not None:
        assert apply_trace_tokens(
            reference_sampled, mapping, traces, lengths, total, prompt
        ) is None
        assert reference_sampled.cpu().tolist() == cpu_expected


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


def test_past_end_of_trace_leaves_sampled_untouched():
    traces = torch.zeros(1, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    traces[0, :2] = _i32([100, 101])
    _assert_replay(
        _i64([999]), _i32([0]), traces, _i32([2]), _i32([6]), _i32([4]), [999]
    )


def test_non_trace_request_untouched():
    traces = torch.zeros(1, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    _assert_replay(
        _i64([42]), _i32([0]), traces, _i32([0]), _i32([3]), _i32([3]), [42]
    )


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
