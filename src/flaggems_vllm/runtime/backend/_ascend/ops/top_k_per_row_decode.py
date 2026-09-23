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

"""Ascend-specialized top_k_per_row_decode (DeepSeek V4 sparse attention).

Decode row i selects from the prefix [0, seq_len[i//next_n] - next_n +
i%next_n + 1) — exactly the prefill op with row_starts == 0 and per-row
ends derived from seq_lens.  This module therefore only materializes those
ranges (one tiny kernel) and forwards to the prefill pipeline, whose
kernels are range-agnostic building blocks (they only read
row_starts/row_ends and know nothing about decode).  The design notes and
measured cost model live in top_k_per_row_prefill.py.
"""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._ascend.ops.top_k_per_row_prefill import (
    _TLE_IMPORT_OK,
)
from flaggems_vllm.runtime.backend._ascend.ops.top_k_per_row_prefill import (
    _top_k_per_row_prefill as _prefill_impl,
)

logger = logging.getLogger(__name__)


@triton.jit
def _ascend_topk_decode_ranges_kernel(
    row_starts,  # out: [num_rows] int32, all zeros
    row_ends,  # out: [num_rows] int32
    seq_lens_ptr,
    next_n,
    num_rows,
    BLOCK: tl.constexpr,
):
    # row i covers [0, seq_lens[i//next_n] - next_n + i%next_n + 1).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < num_rows
    sl = tl.load(seq_lens_ptr + offs // next_n, mask=m, other=0)
    tl.store(row_starts + offs, tl.zeros((BLOCK,), dtype=tl.int32), mask=m)
    tl.store(row_ends + offs, sl - next_n + (offs % next_n) + 1, mask=m)


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for decode phase of DeepSeek V4 sparse attention.

    See flaggems_vllm.ops.top_k_per_row_decode for the argument contract.
    """
    logger.debug("GEMS_ASCEND TOP_K_PER_ROW_DECODE")
    if not _TLE_IMPORT_OK:
        raise NotImplementedError(
            "ascend top_k_per_row_decode requires TLE custom ops "
            "(triton.experimental.tle)"
        )
    device = logits.device
    row_starts = torch.empty(num_rows, dtype=torch.int32, device=device)
    row_ends = torch.empty(num_rows, dtype=torch.int32, device=device)
    _ascend_topk_decode_ranges_kernel[(triton.cdiv(num_rows, 1024),)](
        row_starts, row_ends, seq_lens, next_n, num_rows, BLOCK=1024
    )
    _prefill_impl(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    )
