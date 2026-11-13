# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm

from .megapreattention_ppu_utils import run_tp2

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead" or torch.cuda.device_count() < 2,
    reason="requires two T-Head PPUs",
)


@pytest.mark.megapreattention
@pytest.mark.parametrize(
    "tokens,model_dims,edge,sections",
    [
        (1024, (2048, 16, 2), False, (11, 11, 10)),
        (768, (2048, 16, 4), True, (8, 12, 12)),
        (256, (256, 4, 1), True, (16, 8, 8)),
    ],
)
def test_megapreattention(tokens, model_dims, edge, sections):
    run_tp2(tokens, model_dims=model_dims, edge=edge, sections=sections)


@pytest.mark.megapreattention
def test_megapreattention_replay_after_larger_shape():
    run_tp2(256, larger_tokens=4096)
