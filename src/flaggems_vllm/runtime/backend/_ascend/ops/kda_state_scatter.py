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

"""Fused Triton scatter for the KDA recurrent state cache.

The torch expression used before::

    recurrent_state[state_indices] = result[1].transpose(-1, -2).contiguous().to(dtype)

runs transpose-copy + cast + scatter as separate aclnn kernels with redundant
full-tensor traffic.  This kernel does it in one pass (bitwise vs the chain):

    scatter: cache[idx[n], h, v, k] = cast(final[n, h, k, v])

The recurrent cache stores each slot as ``[H, V, K]`` (K contiguous) while the
AscendC chunk operator produces ``[N, H, K, V]``; the transpose happens inside
the kernel at no extra cost.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["H", "K", "V"])
def _kda_state_scatter_kernel(
    cache,
    state_indices,
    final,
    H,
    K,
    V,
    HKV,
    VK,
    BV: tl.constexpr,
    BK: tl.constexpr,
):
    i_n, i_h, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    idx = tl.load(state_indices + i_n).to(tl.int64)
    if idx < 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m = m_k[:, None] & m_v[None, :]

    b_v = tl.load(
        final + i_n * HKV + i_h * (K * V) + o_k[:, None] * V + o_v[None, :],
        mask=m,
        other=0.0,
    )
    tl.store(
        cache + idx * (H * VK) + i_h * VK + o_v[None, :] * K + o_k[:, None],
        b_v.to(cache.dtype.element_ty),
        mask=m,
    )


def scatter_kda_state(
    cache: torch.Tensor,
    state_indices: torch.Tensor,
    final: torch.Tensor,
) -> None:
    """Scatter ``[N, H, K, V]`` final states back into the ``[S, H, V, K]`` cache.

    Negative slot ids are skipped.  Values are cast to the cache dtype.
    """
    logger.debug("GEMS_ASCEND KDA_STATE_SCATTER")
    n = state_indices.shape[0]
    if n == 0:
        return
    S, H, V, K = cache.shape
    _n, _H, K_, V_ = final.shape
    assert (K_, V_) == (
        K,
        V,
    ), f"final state shape {final.shape} incompatible with cache {cache.shape}"

    BK = triton.next_power_of_2(K)
    BV = 64 if BK >= 64 else triton.next_power_of_2(V)
    _kda_state_scatter_kernel[(n, H, triton.cdiv(V, BV))](
        cache,
        state_indices,
        final,
        H,
        K,
        V,
        H * K * V,
        V * K,
        BV=BV,
        BK=BK,
        num_warps=4,
    )
