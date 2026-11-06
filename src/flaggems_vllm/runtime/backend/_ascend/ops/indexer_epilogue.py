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

"""Fused post-topk epilogue for the GLM lightning indexer.

One program per token row expands the selected pool ids to token ids,
appends the causal tail and writes the int32 output row directly —
replacing the where/full_like/unsqueeze/mul/add/reshape/pad/div/mod/
where/cat/cast chain (~10 small kernels per call).  The history expansion
stays one flat vector (pool id reloaded per lane, 4x redundancy but
cache-resident): a [K, 4] tile has a 4-wide second axis that scalarises on
the Ascend vector cores (measured 139us/row vs ~5us).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["token_start", "topk"])
def _indexer_epilogue_kernel(
    topk_vals_ptr,  # [rows, topk] fp32
    pool_ids_ptr,  # [rows, topk] int64
    positions_ptr,  # global [num_tokens] int64
    output_ptr,  # [num_tokens, 1, POOL_TOPK*KPOOL + KPOOL - 1] int32
    token_start,
    topk,
    POOL_TOPK: tl.constexpr,  # index_topk // index_kpool
    KPOOL: tl.constexpr,  # index_kpool
    HIST_POW2: tl.constexpr,  # next_pow2(POOL_TOPK * KPOOL), for tl.arange
    POOL_OUT: tl.constexpr = False,  # emit pool ids directly (sparse_block_size=KPOOL)
):
    t = tl.program_id(0)
    HIST: tl.constexpr = POOL_TOPK * KPOOL
    jj = tl.arange(0, HIST_POW2 // KPOOL)
    m_j = jj < topk
    # affine [K] loads + one contiguous [K*KPOOL] store: neither the gathered
    # per-lane loads nor a [K, 4] narrow tile vectorise on the Ascend cores
    # (measured 25.5ms vs 0.7ms per 8192-row prefill call).
    vals = tl.load(topk_vals_ptr + t * topk + jj, mask=m_j, other=float("-inf"))
    p = tl.load(pool_ids_ptr + t * topk + jj, mask=m_j, other=0).to(tl.int32)
    p = tl.where((vals != float("-inf")) & m_j, p, -1)
    x = tl.where(p >= 0, p * KPOOL, -1)
    o2048 = tl.interleave(
        tl.interleave(
            tl.where(p >= 0, x + 0, -1),
            tl.where(p >= 0, x + 2, -1),
        ),
        tl.interleave(
            tl.where(p >= 0, x + 1, -1),
            tl.where(p >= 0, x + 3, -1),
        ),
    )
    pos = tl.load(positions_ptr + token_start + t).to(tl.int32)
    if POOL_OUT:
        # pool-id output: [POOL_TOPK pool ids][tail pool id]; the tail pool
        # covers [tail_start, tail_start+KPOOL) and the kernel causally trims
        # tokens past pos.  A full pool boundary (pos+1)%KPOOL==0 has an empty
        # tail -> -1.  -1 must never precede a valid id (kernel breaks on
        # negative pair-leader); topk ordering already puts -1 last.
        out_base_pool = (token_start + t) * (POOL_TOPK + 1)
        tl.store(output_ptr + out_base_pool + jj, p, mask=jj < POOL_TOPK)
        tail_pool = (pos + 1) // KPOOL
        tail_valid = (pos + 1) % KPOOL != 0
        tl.store(
            output_ptr + out_base_pool + POOL_TOPK, tl.where(tail_valid, tail_pool, -1)
        )
    else:
        out_base = (token_start + t) * (HIST + KPOOL - 1)
        offs = tl.arange(0, HIST_POW2)
        tl.store(output_ptr + out_base + offs, o2048, mask=offs < HIST)

        tail_start = (pos + 1) // KPOOL * KPOOL
        tail_count = pos + 1 - tail_start
        ti = tl.arange(0, KPOOL)
        m_ti = ti < KPOOL - 1
        tail = tl.where(ti < tail_count, tail_start + ti, -1)
        tl.store(output_ptr + out_base + HIST + ti, tail, mask=m_ti)


def indexer_epilogue(
    topk_vals: torch.Tensor,
    pool_ids: torch.Tensor,
    positions: torch.Tensor,
    token_start: int,
    pool_topk: int,
    kpool: int,
    pool_out: bool = False,
) -> torch.Tensor:
    """Expand topk pool selections (+ causal tail) into flat int32 rows."""
    logger.debug("GEMS_ASCEND INDEXER_EPILOGUE")
    rows = topk_vals.shape[0]
    topk = topk_vals.shape[1]
    out_width = (pool_topk + 1) if pool_out else pool_topk * kpool + kpool - 1
    out = torch.empty((rows, 1, out_width), dtype=torch.int32, device=topk_vals.device)
    _indexer_epilogue_kernel[(rows,)](
        topk_vals,
        pool_ids,
        positions,
        out,
        token_start,
        topk,
        POOL_TOPK=pool_topk,
        KPOOL=kpool,
        HIST_POW2=triton.next_power_of_2(pool_topk * kpool),
        POOL_OUT=pool_out,
    )
    return out
