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

"""Triton group-list cumsum (per-rank expert token counts -> offsets).

The aclnn cumsum dispatch has no AI_CORE path for integer dtypes on Ascend,
so ``group_list.cumsum(dim=0)`` lands on AI_CPU at ~82us for the typical
EP-per-rank int64[18] shape (measured 1440x per rank in a 16-request
benchmark).  The whole thing is one vector-core program here (~2us device,
~5us amortised against a 62us triton launch wrapper).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["num_experts"])
def _group_list_cumsum_kernel(
    group_list_ptr,
    out_ptr,
    num_experts,
    BLOCK: tl.constexpr,
):
    # Single-program prefix sum over the per-rank expert token counts
    # (num_experts = local_experts, e.g. 288/16 = 18 under EP). The aclnn
    # cumsum dispatch has no AI_CORE path for integer dtypes, so the torch
    # op lands on AI_CPU at ~82us for this shape; the whole thing is one
    # vector-core program here.
    offs = tl.arange(0, BLOCK)
    mask = offs < num_experts
    counts = tl.load(group_list_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, tl.cumsum(counts, axis=0), mask=mask)


def group_list_cumsum(group_list: torch.Tensor) -> torch.Tensor:
    """Prefix-sum the expert token counts in one vector-core program.

    Integer-dtype torch.cumsum dispatches to AI_CPU on Ascend (~82us at the
    EP int64[18] shape); this kernel replaces it at ~5us device time with
    the same launch count.
    """
    logger.debug("GEMS_ASCEND GROUP_LIST_CUMSUM")
    num_experts = group_list.numel()
    out = torch.empty_like(group_list)
    block = triton.next_power_of_2(max(num_experts, 1))
    _group_list_cumsum_kernel[(1,)](group_list, out, num_experts, BLOCK=block)
    return out
