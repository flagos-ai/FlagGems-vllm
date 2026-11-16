# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Correctness tests for the public apply_logit_bias entry point."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

DEVICE = flaggems_vllm.device
NEG_INF = -float("inf")


def _has_device():
    if torch.device(DEVICE).type == "cpu":
        return False
    probe = getattr(torch_device_fn, "is_available", None)
    if probe is None:
        probe = getattr(torch_device_fn, "device_count", None)
    return bool(probe()) if probe is not None else True


pytestmark = [
    pytest.mark.skipif(
        not _has_device(), reason=f"requires an available {DEVICE} device"
    ),
]

DTYPES = [
    torch.float32,
    torch.float16,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(
            not flaggems_vllm.runtime.device.support_bf16,
            reason="backend does not support bfloat16",
        ),
    ),
]


def _i32(data):
    return torch.tensor(data, device=DEVICE, dtype=torch.int32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("check_all_masked_rows", [False, True])
@pytest.mark.apply_logit_bias
def test_apply_logit_bias(dtype, check_all_masked_rows):
    logits = (
        torch.arange(51, device=DEVICE, dtype=torch.float32).reshape(3, 17) / 8
    ).to(dtype)
    mapping = _i32([1, 0, 1])
    pos = _i32([0, 0, 3])
    allowed_count = _i32([0, 2])
    allowed_ids = _i32([[0, 0, 0], [3, 4, 0]])
    bias_count = _i32([2, 2])
    bias_ids = _i32([[1, 7, 0], [3, 6, 0]])
    bias = torch.tensor(
        [[1.0, -2.0, 99.0], [0.5, 3.0, 99.0]], device=DEVICE, dtype=torch.float32
    )
    min_lens = _i32([4, 4])
    stop_count = _i32([1, 2])
    restore = _i32([0, 1])
    stop_ids = _i32([[5, 0, 0], [3, 4, 0]])

    expected = torch.full_like(logits, NEG_INF)
    # Request 0 has no allowlist. Its positive and negative biases stay visible,
    # while its stop token is masked because it is below the minimum length.
    expected[1] = logits[1]
    expected[1, 1] += 1.0
    expected[1, 7] -= 2.0
    expected[1, 5] = NEG_INF

    # At pos + 1 == min_len, stop tokens must not be masked.
    expected[2, 3] = logits[2, 3] + 0.5
    expected[2, 4] = logits[2, 4]

    # Row 0 shares the request with row 2, but is below the threshold.
    # If restoration is enabled, token 3 recovers its post-bias value.
    if check_all_masked_rows:
        expected[0, 3] = logits[0, 3] + 0.5
        expected[0, 4] = logits[0, 4]

    result = flaggems_vllm.apply_logit_bias(
        logits,
        mapping,
        pos,
        allowed_count,
        allowed_ids,
        bias_count,
        bias_ids,
        bias,
        min_lens,
        stop_count,
        restore,
        stop_ids,
        check_all_masked_rows=check_all_masked_rows,
    )

    assert result is None
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.apply_logit_bias
def test_restore_across_vocab_tiles():
    # Index 8192 is in the second tile of the 8192-column row scan.
    vocab_size = 8193
    storage = torch.full((3, vocab_size + 7), 11.0, device=DEVICE, dtype=torch.float32)
    logits = storage[:, :vocab_size]
    logits[:, 0] = 2.0
    logits[:, -1] = NEG_INF
    logits[0, -1] = 7.0

    # Check the row padding as well as the logical vocabulary.
    expected = torch.full_like(storage, 11.0)
    expected[:, :vocab_size] = NEG_INF
    expected[0, vocab_size - 1] = 7.0  # A live non-stop token prevents restoration.
    expected[1, 0] = 2.0  # All masked and this request allows restoration.
    # Row 2 stays fully masked because this request disables restoration.

    zero = _i32([0, 0, 0])
    stop_ids = _i32([[0], [0], [0]])
    flaggems_vllm.apply_logit_bias(
        logits,
        _i32([0, 1, 2]),
        zero,
        _i32([2, 2, 2]),
        _i32([[0, 8192], [0, 8192], [0, 8192]]),
        zero,
        stop_ids,
        torch.zeros((3, 1), device=DEVICE, dtype=torch.float32),
        _i32([2, 2, 2]),
        _i32([1, 1, 1]),
        _i32([1, 1, 0]),
        stop_ids,
        check_all_masked_rows=True,
    )

    torch.testing.assert_close(storage, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "nonfinite",
    [
        pytest.param(float("inf"), id="positive_inf"),
        pytest.param(NEG_INF, id="negative_inf"),
        pytest.param(float("nan"), id="nan"),
    ],
)
@pytest.mark.apply_logit_bias
def test_restore_only_finite_stop_logits(nonfinite):
    logits = torch.tensor([[2.0, nonfinite, NEG_INF]], device=DEVICE, dtype=torch.float32)
    zero = _i32([0])
    ids = _i32([[0, 1]])
    flaggems_vllm.apply_logit_bias(
        logits,
        zero,
        zero,
        zero,
        ids,
        zero,
        ids,
        torch.zeros((1, 2), device=DEVICE, dtype=torch.float32),
        _i32([2]),
        _i32([2]),
        _i32([1]),
        ids,
        check_all_masked_rows=True,
    )
    expected = torch.tensor([[2.0, NEG_INF, NEG_INF]], device=DEVICE, dtype=torch.float32)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("check_all_masked_rows", [False, True])
@pytest.mark.apply_logit_bias
def test_no_stop_token_ids(check_all_masked_rows):
    logits = torch.tensor([[0.25, 1.5]], device=DEVICE, dtype=torch.float32)
    zero = _i32([0])
    one = _i32([1])
    ids = _i32([[1]])
    bias = torch.tensor([[0.5]], device=DEVICE, dtype=torch.float32)

    flaggems_vllm.apply_logit_bias(
        logits,
        zero,
        zero,
        zero,
        ids,
        one,
        ids,
        bias,
        _i32([2]),
        zero,
        one,
        ids,
        check_all_masked_rows=check_all_masked_rows,
    )

    expected = torch.tensor([[0.25, 2.0]], device=DEVICE, dtype=torch.float32)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.apply_logit_bias
def test_apply_logit_bias_empty():
    logits = torch.empty((0, 17), device=DEVICE, dtype=torch.float32)
    mapping = _i32([])
    counts = _i32([0])
    ids = _i32([[0]])
    result = flaggems_vllm.apply_logit_bias(
        logits,
        mapping,
        mapping,
        counts,
        ids,
        counts,
        ids,
        torch.zeros((1, 1), device=DEVICE, dtype=torch.float32),
        counts,
        counts,
        counts,
        ids,
    )

    assert result is None
