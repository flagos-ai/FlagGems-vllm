import random

import pytest
import torch

import flaggems_vllm

from .conftest import QUICK_MODE
from .test_fused_experts_impl import torch_fused_moe_reference

random.seed(42)

FUSED_MOE_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (8, 4, 64, 128, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
]

if not QUICK_MODE:
    FUSED_MOE_CONFIGS += [
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
        (4, 16, 512, 1024, 2),
        # Mixtral-like shapes
        (1, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
        # DeepSeek-V3-like shapes (TP=8 shard)
        (1, 256, 7168, 2048, 8),
        (16, 256, 7168, 2048, 8),
        (64, 256, 7168, 2048, 8),
    ]


def _synchronize():
    if flaggems_vllm.vendor_name == "ascend":
        torch.npu.synchronize()
    elif flaggems_vllm.device == "cuda":
        torch.cuda.synchronize()


def _make_inputs(config, dtype):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flaggems_vllm.device

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * (1.0 / intermediate_size**0.5)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    return hidden_states, w1, w2, topk_weights, topk_ids


@pytest.mark.inplace_fused_experts
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_inplace_fused_experts_accuracy(config, dtype):
    """Test inplace_fused_experts writes correct results into hidden_states."""
    hidden_states, w1, w2, topk_weights, topk_ids = _make_inputs(config, dtype)

    ref = torch_fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids)

    flaggems_vllm.inplace_fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    _synchronize()

    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)
    torch.testing.assert_close(hidden_states, ref, rtol=rtol, atol=atol)


@pytest.mark.inplace_fused_experts
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_inplace_fused_experts_matches_outplace(config, dtype):
    """Test inplace_fused_experts matches outplace_fused_experts."""
    hidden_states, w1, w2, topk_weights, topk_ids = _make_inputs(config, dtype)

    outplace_result = flaggems_vllm.outplace_fused_experts(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    inplace_input = hidden_states.clone()
    flaggems_vllm.inplace_fused_experts(
        inplace_input,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    _synchronize()

    torch.testing.assert_close(inplace_input, outplace_result, rtol=0, atol=0)
