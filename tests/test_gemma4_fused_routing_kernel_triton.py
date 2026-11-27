# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flaggems_vllm


@pytest.mark.gemma4_fused_routing_kernel_triton
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend"
    and (flaggems_vllm.device != "cuda" or not torch.cuda.is_available()),
    reason="requires Ascend or CUDA-compatible tensors",
)
@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [(1, 128, 8), (64, 128, 8), (131, 65, 16), (512, 64, 4), (8192, 128, 8)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gemma4_fused_routing_kernel_triton_accuracy(
    num_tokens, num_experts, topk, dtype
):
    if flaggems_vllm.vendor_name == "ascend":
        from baseline.gemma4_fused_routing_kernel_triton import (
            gemma4_fused_routing_kernel_triton as reference,
        )
    else:
        try:
            from vllm.model_executor.models.gemma4 import (
                gemma4_fused_routing_kernel_triton as reference,
            )
        except ImportError as exc:
            pytest.skip(f"vLLM Gemma 4 routing is unavailable: {exc}")

    device = flaggems_vllm.device
    torch.manual_seed(117)
    logits = torch.randn((num_tokens, num_experts), device=device, dtype=dtype)
    scales = torch.rand(num_experts, device=device, dtype=dtype)

    expected_weights, expected_ids = reference(logits, topk, scales)
    weights, ids = flaggems_vllm.gemma4_fused_routing_kernel_triton(
        logits, topk, scales
    )

    torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=1e-5)


@pytest.mark.gemma4_fused_routing_kernel_triton
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Ascend coverage")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gemma4_fused_routing_ascend_ties_and_padding(dtype):
    from baseline.gemma4_fused_routing_kernel_triton import (
        gemma4_fused_routing_kernel_triton as reference,
    )

    # Non-contiguous inputs, and expert-score ties across the Top-K cutoff.
    logits = torch.empty((65, 3), dtype=dtype).T
    logits.fill_(-1)
    logits[:, 1::2] = 2
    scales = torch.linspace(0.25, 1.25, 130)[::2].to(dtype)
    padding = torch.tensor([False, False, True])
    expected_weights, expected_ids = reference(logits, 7, scales, is_padding=padding)
    npu_logits = logits.T.npu().T
    npu_scales = torch.stack((scales, scales), dim=-1).npu()[:, 0]
    npu_padding = torch.stack((padding, padding), dim=-1).npu()[:, 0]
    weights, ids = flaggems_vllm.gemma4_fused_routing_kernel_triton(
        npu_logits, 7, npu_scales, is_padding=npu_padding
    )
    torch.testing.assert_close(ids.cpu(), expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights.cpu(), expected_weights, atol=1e-6, rtol=1e-5)
