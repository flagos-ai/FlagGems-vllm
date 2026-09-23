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

"""Triton scatter/restore for the KDA short-convolution state cache.

Restores the staged rows of a block-strided conv cache after
``npu_causal_conv1d_custom``.  The previous implementation ran an
``arange/masked_fill`` + ``where/zeros_like/sum`` + ``index_copy_`` sentinel
chain (5+ kernels).  This kernel writes staged rows back in one pass; PAD
entries are skipped, which makes the old row-0 sentinel chain unnecessary
(writing ``cache[0] = cache[0]`` was its only observable effect).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

PAD_SLOT_ID = -1


@triton.jit(do_not_specialize=["slot_numel", "slot_stride", "pad_slot_id"])
def _conv_state_scatter_kernel(
    cache,
    indices,
    staged,
    slot_numel,
    slot_stride,
    pad_slot_id,
    BLOCK: tl.constexpr,
):
    i_n, i_b = tl.program_id(0), tl.program_id(1)
    idx = tl.load(indices + i_n).to(tl.int64)
    if idx == pad_slot_id:
        return

    offs = i_b * BLOCK + tl.arange(0, BLOCK)
    m = offs < slot_numel
    b_v = tl.load(staged + i_n * slot_numel + offs, mask=m, other=0)
    tl.store(cache + idx * slot_stride + offs, b_v, mask=m)


def scatter_conv_state(
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    staged: torch.Tensor,
) -> None:
    """Write staged conv rows back; PAD entries are skipped."""
    logger.debug("GEMS_ASCEND KDA_CONV_SCATTER")
    flat_indices = cache_indices.reshape(-1)
    n = flat_indices.shape[0]
    if n == 0:
        return
    slot_numel = conv_state[0].numel()
    BLOCK = max(2048, triton.next_power_of_2(triton.cdiv(slot_numel, 4)))
    _conv_state_scatter_kernel[(n, triton.cdiv(slot_numel, BLOCK))](
        conv_state,
        flat_indices,
        staged,
        slot_numel,
        conv_state.stride(0),
        PAD_SLOT_ID,
        BLOCK=BLOCK,
        num_warps=8,
    )
