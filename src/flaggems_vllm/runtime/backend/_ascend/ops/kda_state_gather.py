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

"""Fused Triton gather for the KDA recurrent state cache.

The torch expression used before::

    initial_state_vk = recurrent_state[state_indices].contiguous()
    clear_ssm_states(initial_state_vk, has_initial_state)
    initial_state_kv = initial_state_vk.transpose(-1, -2).contiguous()

round-trips the state through gather -> zero -> transpose-copy (multiple aclnn
kernels, redundant full-tensor reads/writes).  This kernel does it in one pass:

    gather: out[n, h, k, v] = has_initial_state[n] ? cache[idx[n], h, v, k] : 0

The recurrent cache stores each slot as ``[H, V, K]`` (K contiguous) while the
AscendC chunk operator expects ``[N, H, K, V]``; the transpose happens inside
the kernel at no extra cost.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["H", "K", "V"])
def _kda_state_gather_kernel(
    cache,
    state_indices,
    has_initial_state,
    out,
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
    has_state = tl.load(has_initial_state + i_n).to(tl.int1)
    valid = (idx >= 0) & has_state

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m = m_k[:, None] & m_v[None, :]

    # Safe row for invalid entries; the result is zeroed via tl.where below.
    src_row = tl.where(idx >= 0, idx, 0)
    b_v = tl.load(
        cache + src_row * (H * VK) + i_h * VK + o_v[None, :] * K + o_k[:, None],
        mask=m,
        other=0.0,
    ).to(out.dtype.element_ty)
    b_v = tl.where(valid, b_v, tl.zeros_like(b_v))
    tl.store(
        out + i_n * HKV + i_h * (K * V) + o_k[:, None] * V + o_v[None, :], b_v, mask=m
    )


def gather_kda_state(
    cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Gather recurrent states into ``[N, H, K, V]`` (zeroed where no initial state).

    Args:
        cache: ``[S, H, V, K]`` recurrent state cache (fp32 or bf16).
        state_indices: ``[N]`` slot id per sequence.
        has_initial_state: ``[N]`` bool/int flag per sequence.

    Returns:
        ``[N, H, K, V]`` tensor in cache dtype with the last two dims
        transposed to the AscendC chunk layout.
    """
    logger.debug("GEMS_ASCEND KDA_STATE_GATHER")
    n = state_indices.shape[0]
    if n == 0:
        return cache[:0].transpose(-1, -2).contiguous()
    S, H, V, K = cache.shape
    out = torch.empty((n, H, K, V), dtype=cache.dtype, device=cache.device)

    BK = triton.next_power_of_2(K)
    BV = 64 if BK >= 64 else triton.next_power_of_2(V)
    _kda_state_gather_kernel[(n, H, triton.cdiv(V, BV))](
        cache,
        state_indices,
        has_initial_state,
        out,
        H,
        K,
        V,
        H * K * V,
        V * K,
        BV=BV,
        BK=BK,
        num_warps=4,
    )
    return out
