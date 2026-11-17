# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import triton
import triton.language as tl

import flaggems_vllm

from .megapreattention_utils import run_distributed

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead" or torch.cuda.device_count() == 0,
    reason="requires a T-Head PPU",
)


@pytest.mark.megapreattention
@pytest.mark.parametrize(
    "tp,tokens,model_dims,edge,sections",
    [
        (2, 1024, (2048, 16, 2), False, (11, 11, 10)),
        (2, 768, (2048, 16, 4), True, (8, 12, 12)),
        (2, 256, (256, 4, 1), True, (16, 8, 8)),
        (3, 384, (256, 6, 3), False, (11, 11, 10)),
        (4, 2048, (256, 8, 4), True, (8, 12, 12)),
        (8, 4096, (256, 16, 2), True, (16, 8, 8)),
        (16, 2048, (256, 16, 2), True, (16, 8, 8)),
        (1, 1, (256, 4, 2), True, (11, 11, 10)),
        (2, 34, (320, 8, 2), True, (16, 8, 8)),
        (2, 128, (256, 4, 1), False, (11, 11, 10)),
        (3, 387, (256, 6, 3), False, (16, 8, 8)),
        (4, 1036, (320, 8, 4), True, (8, 12, 12)),
        (8, 520, (256, 16, 2), True, (16, 8, 8)),
    ],
)
def test_megapreattention(tp, tokens, model_dims, edge, sections):
    if torch.cuda.device_count() < tp:
        pytest.skip(f"requires {tp} T-Head PPUs")
    from flaggems_vllm.runtime.backend._thead.ops import megapreattention

    assert flaggems_vllm.megapreattention is megapreattention
    assert flaggems_vllm.ops.__all__.count("megapreattention") == 1
    run_distributed(tokens, tp=tp, model_dims=model_dims, edge=edge, sections=sections)


@pytest.mark.megapreattention
@pytest.mark.parametrize("tokens", [34, 256])
def test_megapreattention_replay_after_larger_shape(tokens):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two T-Head PPUs")
    run_distributed(tokens, larger_tokens=8192)


@pytest.mark.megapreattention
def test_megapreattention_large_cache_offsets():
    from flaggems_vllm.runtime.backend._thead.ops._megapreattention_kernel import (
        _cache_offset,
    )

    @triton.jit
    def probe(output, slot):
        head = tl.program_id(0)
        offset = _cache_offset(slot, head, 1 << 32, 1 << 30, 512, 1 << 21)
        tl.store(output + head, offset)

    output = torch.empty(4, dtype=torch.int64, device="cuda")
    probe[(4,)](output, (1 << 21) + 7)
    assert output.tolist() == [(1 << 32) + (head << 30) + 7 * 512 for head in range(4)]
