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

"""Accuracy tests for top_k_top_p (Ascend sampling-filter kernel).

The kernel applies per-row top-k / top-p filtering to ``[batch, vocab]``
logits: masked positions become ``-inf``, kept positions keep their original
logit values.  Because tie-breaking between equally-sized probabilities is not
specified, the comparison is mask-based (the finite/unfinite pattern must
match exactly) plus a value check on the commonly kept positions, following
the original operator's test methodology.
"""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._ascend.ops.top_k_top_p import top_k_top_p

from .conftest import QUICK_MODE

torch_npu = pytest.importorskip("torch_npu")

device = flaggems_vllm.device

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized top_k_top_p targets Ascend",
)

if QUICK_MODE:
    SHAPE_LIST = [(1, 256)]
    TOP_K_LIST = [5]
else:
    SHAPE_LIST = [(1, 256), (2, 4096), (4, 8192)]
    TOP_K_LIST = [None, 5, 20]


def top_k_top_p_ref(
    logits: torch.Tensor,
    p: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch reference of the top-k + top-p sampling filter."""
    if p is None and k is None:
        raise ValueError("p and k cannot both be None")
    if logits.dim() != 2:
        raise ValueError("logits must be 2D [batch, vocab]")
    batch, vocab = logits.shape
    if p is not None and p.shape != (batch,):
        raise ValueError("p must have shape [batch]")
    if k is not None and k.shape != (batch,):
        raise ValueError("k must have shape [batch]")

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


@pytest.mark.top_k_top_p
@pytest.mark.parametrize("shape", SHAPE_LIST)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("top_k", TOP_K_LIST)
@pytest.mark.parametrize("top_p", [0.90])
def test_top_k_top_p(shape, dtype, top_k, top_p):
    logits = torch.randn(shape, device=device, dtype=dtype)
    p = torch.full((shape[0],), top_p, device=device, dtype=torch.float32)
    k = (
        None
        if top_k is None
        else torch.full((shape[0],), top_k, device=device, dtype=torch.int32)
    )
    actual = top_k_top_p(logits.clone(), p=p, k=k)
    expected = top_k_top_p_ref(logits.clone(), p=p, k=k)
    actual_mask = torch.isfinite(actual)
    expected_mask = torch.isfinite(expected)
    assert int((actual_mask ^ expected_mask).sum().item()) == 0
    common = actual_mask & expected_mask
    torch.testing.assert_close(actual[common], expected[common], atol=2e-3, rtol=2e-3)
