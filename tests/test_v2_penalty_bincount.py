# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm


@pytest.mark.v2_penalty_bincount
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "kunlunxin", reason="Kunlunxin required"
)
@pytest.mark.parametrize(
    "vocab,max_prefill_len", [(167, 185), (151939, 185), (151939, 2048), (167, 0)]
)
def test_v2_penalty_bincount(vocab, max_prefill_len):
    generator = torch.Generator().manual_seed(147)
    ids = torch.randint(
        vocab, (4, max(190, max_prefill_len)), dtype=torch.int32, generator=generator
    )
    ids[:, :4] = torch.tensor([31, 31, 12, vocab - 1])
    mapping = torch.tensor([3, 1], dtype=torch.int64)
    prompt_len = torch.tensor([20, 135, 0, 72], dtype=torch.int32)
    prefill_len = torch.tensor([44, 185, 0, 175], dtype=torch.int32)
    if max_prefill_len == 2048:
        prompt_len[:] = 513
        prefill_len[:] = max_prefill_len
        # Repeated output IDs span several counting tiles. P800 can lose
        # increments if different programs update the same output bin.
        ids[:, 520::133] = 31
        ids[:, 670::217] = vocab - 1
    elif max_prefill_len == 0:
        prompt_len.zero_()
        prefill_len.zero_()

    words = (vocab + 31) // 32
    packed = torch.full((4, words + 2), 7, dtype=torch.int32)
    counts = torch.full((4, vocab + 5), 7, dtype=torch.int32)
    expected_packed = packed.clone()
    expected_counts = counts.clone()
    for row in mapping.tolist():
        expected_packed[row, :words] = 0
        expected_counts[row, :vocab] = 0
        for position, token in enumerate(ids[row, : prefill_len[row]].tolist()):
            if position < prompt_len[row]:
                bit = 1 << (token % 32)
                expected_packed[row, token // 32] |= (
                    bit if bit < 1 << 31 else -(1 << 31)
                )
            else:
                expected_counts[row, token] += 1

    packed_gpu = packed.cuda()
    counts_gpu = counts.cuda()
    flaggems_vllm.v2_penalty_bincount(
        mapping.cuda(),
        ids.cuda(),
        prompt_len.cuda(),
        prefill_len.cuda(),
        packed_gpu[:, :words],
        counts_gpu[:, :vocab],
        max_prefill_len,
    )
    torch.testing.assert_close(packed_gpu.cpu(), expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(counts_gpu.cpu(), expected_counts, rtol=0, atol=0)
