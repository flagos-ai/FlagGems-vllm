# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Adversarial two-key cancellation across split-KV boundaries."""
import math

import pytest
import torch

import flaggems_vllm as gems
from tests.accuracy_utils import gems_assert_close

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8,
    pytest.mark.skipif(gems.vendor_name != "metax", reason="MetaX decode residuals"),
]


@pytest.mark.parametrize("batch", [17, 512])
@pytest.mark.parametrize("length", [32, 2051, 65536])
@pytest.mark.parametrize("scale", [1.7, 16.0, -16.0])
def test_decode_residual_cancellation(length, scale, batch):
    heads, dim, page = 4, 256, 16
    npages = (length + page - 1) // page
    q = torch.full((batch, heads, dim), 127, device=gems.device, dtype=torch.int8)
    q[:, :, 0] = 1
    k = torch.full((npages, page, 1, dim), -127, device=gems.device, dtype=torch.int8)
    v = torch.zeros_like(k)
    k[0, 0] = 0
    k[(length - 1) // page, (length - 1) % page] = 0
    k[(length - 1) // page, (length - 1) % page, 0, 0] = 1
    v[0, 0] = 127
    v[(length - 1) // page, (length - 1) % page] = -127
    table = torch.arange(npages, device=gems.device, dtype=torch.int32)[None, :].expand(
        batch, npages
    )
    used = torch.full((batch,), length, device=gems.device, dtype=torch.int32)
    cu = torch.arange(batch + 1, device=gems.device, dtype=torch.int32)
    qs = torch.ones((batch, heads, 1), device=gems.device)
    gap = 0.000244
    ks = torch.full((batch, 1, 1), gap * math.sqrt(dim), device=gems.device).expand(
        batch, 1, 576
    )
    vs = torch.full((batch, 1, 1), scale, device=gems.device).expand(batch, 1, 576)
    actual, lse = gems.flash_attn_varlen_func(
        q,
        k,
        v,
        1,
        cu,
        73728,
        seqused_k=used,
        block_table=table,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        causal=True,
        num_splits=32,
        return_softmax_lse=True,
    )
    expected = torch.full_like(actual.float(), -127 * scale * math.tanh(gap / 2))
    gems_assert_close(actual.float(), expected, dtype=torch.float32, atol=0.025)
    expected_lse = torch.full_like(lse, math.log1p(math.exp(gap)))
    gems_assert_close(lse, expected_lse, dtype=torch.float32, atol=0.0015)
