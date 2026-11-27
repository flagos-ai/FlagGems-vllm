# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm
from tests.lightning_attention_utils import (
    MINIMAX_DECODE_BATCHES,
    assert_relative,
    decode_reference,
    make_decode_case,
    make_initial,
    make_qkv,
    make_slope,
)

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend", reason="requires Ascend"
)


@pytest.mark.lightning_attention_decode
@pytest.mark.parametrize("batch", MINIMAX_DECODE_BATCHES)
def test_lightning_attention_decode(batch):
    torch.manual_seed(932)
    qkv, slope, cache, slots, initial = make_decode_case(batch)
    active, sentinel = cache[-batch:], cache[-batch - 1]
    active.copy_(initial)
    sentinel.fill_(0.25)
    expected_state = initial.cpu()
    reference_qkv = tuple(x.cpu() for x in qkv)
    reference_slope, reference_slots = slope.cpu(), torch.arange(batch)
    for _ in range(3):
        expected = decode_reference(
            *reference_qkv,
            expected_state,
            reference_slope,
            reference_slots,
            dtype=torch.float64,
        )
        result = flaggems_vllm.lightning_attention_decode(*qkv, cache, slope, slots)
        assert result.dtype == expected.dtype
        assert_relative(result.cpu(), expected, 2e-4)
        assert_relative(active.cpu(), expected_state, 2e-4)
    assert torch.all(sentinel == 0.25).item()


@pytest.mark.lightning_attention_decode
def test_lightning_attention_decode_padding_and_strides():
    qkv = tuple(x.transpose(1, 3).contiguous().transpose(1, 3) for x in make_qkv(3, 1))
    slope = make_slope()
    cache = make_initial(4, torch.float32).transpose(-1, -2)
    slots = torch.tensor([2, -1, 0], dtype=torch.int64, device="npu")
    state = cache.cpu().clone()
    expected = decode_reference(
        *(x.cpu() for x in qkv), state, slope.cpu(), slots.cpu(), dtype=torch.float64
    )
    actual = flaggems_vllm.lightning_attention_decode(*qkv, cache, slope, slots)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(cache.cpu(), state, rtol=1e-3, atol=1e-3)
    assert torch.count_nonzero(actual[1]).item() == 0


@pytest.mark.lightning_attention_decode
def test_lightning_attention_decode_empty_batch():
    result = flaggems_vllm.lightning_attention_decode(
        *make_qkv(0, 1),
        make_initial(1),
        make_slope(),
        torch.empty(0, dtype=torch.int32, device="npu"),
    )
    assert result.shape == (0, 1024)
