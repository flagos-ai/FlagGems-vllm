# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm
from benchmark.test_lightning_attention import run_pair
from tests.lightning_attention_utils import MINIMAX_DECODE_BATCHES, make_decode_case

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend", reason="requires Ascend"
)


@pytest.mark.lightning_attention_decode
@pytest.mark.parametrize("batch", MINIMAX_DECODE_BATCHES)
def test_lightning_attention_decode_benchmark(batch, request):
    native = pytest.importorskip("vllm.model_executor.layers.lightning_attn")
    torch.manual_seed(932)
    qkv, slope, cache, slots, initial = make_decode_case(batch)
    active = cache[-batch:]
    run_pair(
        request,
        "lightning_attention_decode",
        [batch, 8, 1, 128],
        {
            "vllm": lambda: native.linear_decode_forward_triton(
                *qkv, cache, slope, slots
            ),
            "gems": lambda: flaggems_vllm.lightning_attention_decode(
                *qkv, cache, slope, slots
            ),
        },
        lambda: active.copy_(initial),
        32,
        11,
    )
