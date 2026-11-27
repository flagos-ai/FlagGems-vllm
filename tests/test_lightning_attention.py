# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm
from tests.lightning_attention_utils import (
    MINIMAX_PREFIX_LENGTHS,
    make_initial,
    make_qkv,
    make_slope,
    prefix_reference,
)

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend", reason="requires Ascend"
)


def check_prefill(qkv, slope, initial):
    before = initial.cpu().clone() if initial is not None else None
    output, history = flaggems_vllm.lightning_attention(*qkv, slope, kv_history=initial)
    expect_out, expect_history = prefix_reference(
        *(x.cpu() for x in qkv), slope.cpu(), before
    )
    assert output.dtype == torch.float32 and history.dtype == torch.float32
    if before is not None:
        expect_history[:, :, -1] = expect_history[:, :, -1].to(initial.dtype)
        torch.testing.assert_close(initial.cpu(), before, rtol=0, atol=0)
    torch.testing.assert_close(output.cpu().double(), expect_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        history.cpu().double(), expect_history, rtol=1e-3, atol=1e-3
    )


@pytest.mark.lightning_attention
@pytest.mark.parametrize("tokens", MINIMAX_PREFIX_LENGTHS)
def test_lightning_attention(tokens):
    torch.manual_seed(932)
    check_prefill(make_qkv(1, tokens), make_slope(), make_initial(1))


@pytest.mark.lightning_attention
@pytest.mark.parametrize("tokens", [1, 256, 257, 8064, 8065])
def test_lightning_attention_dispatch_and_layout(tokens):
    qkv = tuple(
        x.transpose(1, 2).contiguous().transpose(1, 2) for x in make_qkv(1, tokens)
    )
    check_prefill(qkv, make_slope(), None)


@pytest.mark.lightning_attention
def test_lightning_attention_rejects_wrong_block():
    qkv = make_qkv(1, 2)
    with pytest.raises(NotImplementedError, match="block_size=256"):
        flaggems_vllm.lightning_attention(*qkv, make_slope(), block_size=128)
