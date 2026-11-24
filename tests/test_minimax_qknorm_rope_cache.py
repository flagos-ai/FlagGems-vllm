# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import flaggems_vllm
from baseline.minimax_qknorm_rope_cache import build_case, check_call

pytestmark = [
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "ascend", reason="Ascend implementation"
    ),
]


@pytest.fixture(autouse=True)
def npu_device():
    device = torch.device("npu", int(os.environ.get("MINIMAX_DEVICE", "0")))
    with torch.npu.device(device):
        if not torch.npu.get_device_name(device).startswith("Ascend910B"):
            pytest.skip("Ascend 910B correctness tests")
        previous = torch.npu.config.allow_internal_format
        torch.npu.config.allow_internal_format = False
        try:
            yield device
        finally:
            torch.npu.config.allow_internal_format = previous


@pytest.mark.parametrize(
    "tokens,heads", [(0, 8), (1, 8), (3, 16), (16, 8), (17, 16), (128, 8)]
)
@pytest.mark.parametrize("layout", ["compact", "padded"])
@pytest.mark.minimax_qknorm_rope_cache
@torch.inference_mode()
def test_minimax_qknorm_rope_cache(tokens, heads, layout, npu_device):
    case = build_case(tokens, heads, npu_device, layout=layout, edge=tokens != 1)
    check_call(flaggems_vllm.minimax_qknorm_rope_cache, case)


@pytest.mark.minimax_qknorm_rope_cache
@torch.inference_mode()
def test_minimax_rejects_unsupported_heads(npu_device):
    kwargs, _, _ = build_case(1, 8, npu_device, layout="compact")
    kwargs["num_q_heads"] = 4
    with pytest.raises(NotImplementedError, match="Q8 or Q16"):
        flaggems_vllm.minimax_qknorm_rope_cache(**kwargs)


@pytest.mark.minimax_qknorm_rope_cache
@torch.inference_mode()
def test_minimax_rejects_int32_positions(npu_device):
    kwargs, _, _ = build_case(1, 8, npu_device, layout="compact")
    kwargs["positions"] = kwargs["positions"].int()
    with pytest.raises(ValueError, match="positions"):
        flaggems_vllm.minimax_qknorm_rope_cache(**kwargs)
