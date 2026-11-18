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

import random
import time

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import topk_softmax

from . import accuracy_utils as utils

# Make sure every thread has same seed.
random.seed(time.time() // 100)


def generate_test_params():
    params = [torch.int32, torch.int64]
    if utils.SkipVersion("torch", ">2.2"):
        params.append(torch.uint32)

    return params


@pytest.mark.skipif(
    flaggems_vllm.vendor_name == "metax", reason="Issue #2857: RuntimeError"
)
@pytest.mark.topk_softmax
@pytest.mark.parametrize("index_dtype", generate_test_params())
@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    [
        (1, 4, 2),
        (4, 8, 2),
        (8, 16, 4),
        (32, 64, 8),
        (128, 128, 16),
        (500, 255, 30),
        (512, 256, 32),
        (1024, 512, 32),
    ],
)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax(
    num_tokens, num_experts, topk, input_dtype, index_dtype, renormalize
):
    if flaggems_vllm.vendor_name == "mthreads" and index_dtype == torch.uint32:
        # Issue #2858: torch musa does not support uint32
        index_dtype = torch.int64

    try:
        from vllm._custom_ops import topk_softmax as vllm_topk_softmax
    except (ImportError, AttributeError):
        pytest.skip("vLLM topk_softmax not available")

    torch.manual_seed(42)
    device = flaggems_vllm.device

    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    vllm_weights = torch.empty(num_tokens, topk, device=device, dtype=torch.float32)
    vllm_indices = torch.empty(num_tokens, topk, device=device, dtype=index_dtype)
    vllm_token_expert = torch.empty(num_tokens, topk, device=device, dtype=torch.int32)

    vllm_topk_softmax(
        vllm_weights,
        vllm_indices,
        vllm_token_expert,
        gating_output,
        renormalize,
    )

    gems_weights = torch.empty_like(vllm_weights)
    gems_indices = torch.empty_like(vllm_indices)
    gems_token_expert = torch.empty_like(vllm_token_expert)

    topk_softmax(
        gems_weights,
        gems_indices,
        gems_token_expert,
        gating_output,
        renormalize,
    )

    assert torch.allclose(
        gems_weights, vllm_weights, atol=1e-5
    ), "topk_weights mismatch"
    assert torch.equal(
        gems_indices.cpu(), vllm_indices.cpu()
    ), "topk_indices mismatch (fp32)"
    assert torch.equal(
        gems_token_expert.cpu(), vllm_token_expert.cpu()
    ), "token_expert_indices mismatch"

    if renormalize:
        sums = gems_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@pytest.mark.topk_softmax
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("expert_count", [5, 256, 257, 512])
def test_topk_softmax_bias_and_padding(normalize, dtype, expert_count):
    logits = torch.tensor(
        [
            [2, 1, 0, 1, 4],
            [4, 3, 2, 1, 0],
            [1, 3, 2, 3, 0],
            [float("nan"), 1, 2, 3, 4],
            [float("inf"), 1, 2, 3, 4],
            [float("-inf")] * 5,
        ],
        device="cuda",
        dtype=dtype,
    )
    bias = torch.tensor([0.0, 0.75, 0.0, 0.75, -0.5], device="cuda")
    if expert_count > logits.shape[1]:
        extra = torch.arange(expert_count - logits.shape[1], device="cuda")
        extra_logits = (extra.remainder(13).float() / 5 - 2).to(dtype)
        logits = torch.cat((logits, extra_logits.expand(logits.shape[0], -1)), dim=1)
        bias = torch.cat((bias, (extra.remainder(7).float() - 3) * 0.03125))
    padding = torch.tensor([False, True, False, False, False, False], device="cuda")
    weights = torch.empty((6, 3), device="cuda")
    indices = torch.empty((6, 3), device="cuda", dtype=torch.int64)
    source = torch.empty_like(indices, dtype=torch.int32)

    topk_softmax(weights, indices, source, logits, normalize, bias, padding)

    probabilities = torch.softmax(logits.float(), dim=-1).nan_to_num()
    ranking = torch.argsort(probabilities + bias, dim=-1, descending=True, stable=True)
    expected_indices = ranking[:, :3]
    expected_weights = probabilities.gather(1, expected_indices)
    if normalize:
        denominator = expected_weights.sum(dim=-1, keepdim=True)
        expected_weights /= torch.where(denominator > 0, denominator, 1.0)
    expected_source = (
        torch.arange(3, device="cuda", dtype=torch.int32)[None, :] * 6
        + torch.arange(6, device="cuda", dtype=torch.int32)[:, None]
    )
    torch.testing.assert_close(
        weights, expected_weights.masked_fill(padding[:, None], 0)
    )
    torch.testing.assert_close(
        indices, expected_indices.masked_fill(padding[:, None], -1)
    )
    torch.testing.assert_close(source, expected_source)


@pytest.mark.topk_softmax
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "metax",
    reason="MetaX CUDA runtime required",
)
def test_topk_softmax_metax_unpadded_without_bias():
    logits = torch.arange(9 * 256, device="cuda", dtype=torch.float32).reshape(9, 256)
    logits = logits.remainder(19) * 0.25
    logits[0, 0] = float("nan")
    weights = torch.empty((9, 8), device="cuda")
    ids = torch.empty((9, 8), device="cuda", dtype=torch.int32)
    source = torch.empty_like(ids)
    topk_softmax(weights, ids, source, logits, True)

    probabilities = torch.softmax(logits, dim=-1).nan_to_num()
    expected_ids = probabilities.argsort(dim=1, descending=True, stable=True)[:, :8]
    expected_weights = probabilities.gather(1, expected_ids)
    denominator = expected_weights.sum(dim=1, keepdim=True)
    expected_weights /= torch.where(denominator > 0, denominator, 1.0)
    expected_source = (
        torch.arange(8, device="cuda", dtype=torch.int32)[None, :] * 9
        + torch.arange(9, device="cuda", dtype=torch.int32)[:, None]
    )
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(source, expected_source)
