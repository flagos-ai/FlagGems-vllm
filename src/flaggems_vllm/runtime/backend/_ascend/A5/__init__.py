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
"""Operators specific to the Ascend A5 generation (950 series, arch35).

These are kept apart from ``_ascend/FLA`` because they are tuned against
A5's unified-buffer budget and its 64 vector / 32 cube core split, and are
not expected to be the right choice on earlier Ascend parts.
"""

from flaggems_vllm.runtime.backend._ascend.A5.chunk_gated_delta_rule_fwd_intra_npu import (
    chunk_gated_delta_rule_fwd_intra_npu_unfused,
    chunk_scaled_dot_kkt_fwd_950dt,
    recompute_w_u_fwd_950dt,
    solve_tril_950dt,
)
from flaggems_vllm.runtime.backend._ascend.A5.chunk_gated_delta_rule_fwd_intra_npu_fused import (
    chunk_gated_delta_rule_fwd_intra_npu_fused,
)
from flaggems_vllm.runtime.backend._ascend.A5.gdn_chunk_recurrent_950dt import (
    chunk_gdn_950dt,
    fused_recurrent_gdn_950dt,
)

__all__ = [
    "chunk_gated_delta_rule_fwd_intra_npu_fused",
    "chunk_gated_delta_rule_fwd_intra_npu_unfused",
    "chunk_gdn_950dt",
    "chunk_scaled_dot_kkt_fwd_950dt",
    "fused_recurrent_gdn_950dt",
    "recompute_w_u_fwd_950dt",
    "solve_tril_950dt",
]
