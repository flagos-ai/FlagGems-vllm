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

"""Triton gather/staging for the KDA short-convolution state cache.

``npu_causal_conv1d_custom`` needs a contiguous cache, so when the block-strided
pool cannot be used directly the rows used by this batch must be staged in and
out.  The previous implementation staged with ``index_select().contiguous()`` +
``arange/masked_fill`` (multiple kernels per call).  This kernel does the same
job in one pass: copy the valid rows (invalid rows alias row 0, exactly like
the old ``masked_fill(~valid, 0)``) and emit local row indices with PAD markers.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

PAD_SLOT_ID = -1


@triton.jit(do_not_specialize=["slot_numel", "slot_stride", "pad_slot_id"])
def _conv_state_gather_kernel(
    cache,
    indices,
    staged,
    local_indices,
    slot_numel,
    slot_stride,
    pad_slot_id,
    BLOCK: tl.constexpr,
):
    i_n, i_b = tl.program_id(0), tl.program_id(1)
    idx = tl.load(indices + i_n).to(tl.int64)
    valid = idx != pad_slot_id

    offs = i_b * BLOCK + tl.arange(0, BLOCK)
    m = offs < slot_numel
    src_row = tl.where(valid, idx, 0)
    b_v = tl.load(cache + src_row * slot_stride + offs, mask=m, other=0)
    tl.store(staged + i_n * slot_numel + offs, b_v, mask=m)
    if i_b == 0:
        tl.store(local_indices + i_n, tl.where(valid, i_n, pad_slot_id))


def gather_conv_state(
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage the selected rows of a block-strided conv cache contiguously.

    Returns ``(staged, local_indices)`` where ``staged[i]`` is a copy of
    ``conv_state[cache_indices[i]]`` (row 0 for PAD entries) and
    ``local_indices[i]`` is ``i`` or ``PAD_SLOT_ID``.
    """
    logger.debug("GEMS_ASCEND KDA_CONV_GATHER")
    flat_indices = cache_indices.reshape(-1)
    n = flat_indices.shape[0]
    slot_numel = conv_state[0].numel()
    staged = torch.empty(
        (n, *conv_state.shape[1:]), dtype=conv_state.dtype, device=conv_state.device
    )
    local_indices = torch.empty_like(flat_indices)

    BLOCK = max(2048, triton.next_power_of_2(triton.cdiv(slot_numel, 4)))
    _conv_state_gather_kernel[(n, triton.cdiv(slot_numel, BLOCK))](
        conv_state,
        flat_indices,
        staged,
        local_indices,
        slot_numel,
        conv_state.stride(0),
        PAD_SLOT_ID,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return staged, local_indices.view_as(cache_indices)
