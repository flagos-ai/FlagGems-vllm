# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm

from .megapreattention_utils import run_distributed

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead" or torch.cuda.device_count() < 2,
    reason="requires two T-Head PPUs",
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
def test_megapreattention_replay_after_larger_shape():
    run_distributed(256, larger_tokens=4096)
