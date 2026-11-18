# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm


@pytest.mark.gemma4_fused_routing_kernel_triton
@pytest.mark.skipif(
    flaggems_vllm.device != "cuda" or not torch.cuda.is_available(),
    reason="requires CUDA-compatible tensors",
)
@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [(1, 128, 8), (64, 128, 8), (131, 65, 16), (512, 64, 4), (8192, 128, 8)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gemma4_fused_routing_kernel_triton_accuracy(
    num_tokens, num_experts, topk, dtype
):
    try:
        from vllm.model_executor.models.gemma4 import (
            gemma4_fused_routing_kernel_triton as vllm_reference,
        )
    except ImportError as exc:
        pytest.skip(f"vLLM Gemma 4 routing is unavailable: {exc}")

    device = flaggems_vllm.device
    torch.manual_seed(117)
    logits = torch.randn((num_tokens, num_experts), device=device, dtype=dtype)
    scales = torch.rand(num_experts, device=device, dtype=dtype)

    expected_weights, expected_ids = vllm_reference(logits, topk, scales)
    weights, ids = flaggems_vllm.gemma4_fused_routing_kernel_triton(
        logits, topk, scales
    )

    torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=1e-5)
