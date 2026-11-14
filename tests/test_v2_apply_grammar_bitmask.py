# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm


@pytest.mark.v2_apply_grammar_bitmask
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "kunlunxin", reason="Kunlunxin required"
)
@pytest.mark.parametrize(
    "vocab,dtype", [(167, torch.float32), (151939, torch.bfloat16)]
)
def test_v2_apply_grammar_bitmask(vocab, dtype):
    generator = torch.Generator().manual_seed(147)
    data = torch.randn((4, vocab + 3), generator=generator).to(dtype)
    mask = torch.randint(
        -(1 << 31),
        1 << 31,
        (5, (vocab + 31) // 32),
        generator=generator,
        dtype=torch.int32,
    )
    mapping = torch.tensor([0, 2, 0, 3, 5], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 3, 4], dtype=torch.int32)
    token = torch.arange(vocab)
    blocked = ((mask[:, token // 32] >> (token % 32)) & 1) == 0
    expected = data.clone()
    for i, index in enumerate(mapping.tolist()):
        row, position = divmod(index, 2)
        if position < offsets[row + 1] - offsets[row]:
            expected[offsets[row] + position, :vocab].masked_fill_(
                blocked[i], -torch.inf
            )

    device_data = data.cuda()
    flaggems_vllm.v2_apply_grammar_bitmask(
        device_data[:, :vocab], mask.cuda(), mapping.cuda(), offsets.cuda(), 2
    )
    torch.testing.assert_close(device_data.cpu(), expected, rtol=0, atol=0)
