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

"""Prepared KDA H recurrence with BF16 snapshots and FP32 V-first state."""
import logging

from flaggems_vllm import runtime

logger = logging.getLogger(__name__)


def chunk_kda_fwd_h(k, w, u, *, gk, initial_state, cu_seqlens, value_tile=128):
    """Return ``(snapshots, v_new, final_state)`` for 64-token KDA chunks.

    ``k/w/u`` are BF16 ``[1,H,T,128]``; ``gk`` is FP32 base-2 chunk-local
    cumulative gates. ``initial_state`` is FP32 ``[N,H,V,K]``. ``cu_seqlens``
    is a host list/tuple, not a device tensor. Snapshots are BF16
    ``[1,total_chunks,H,V,K]``; final state is FP32 and inputs are unmodified.
    Ascend only, inference only. Uses fixed validated tiles without autotuning.
    """
    if runtime.device.vendor_name != "ascend":
        raise NotImplementedError("chunk_kda_fwd_h requires the Ascend CANN backend")
    from flaggems_vllm.runtime.backend._ascend.ops.chunk_kda_fwd_h import (
        chunk_kda_fwd_h as impl,
    )

    logger.debug("GEMS CHUNK_KDA_FWD_H")
    return impl(
        k,
        w,
        u,
        gk=gk,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        value_tile=value_tile,
    )
