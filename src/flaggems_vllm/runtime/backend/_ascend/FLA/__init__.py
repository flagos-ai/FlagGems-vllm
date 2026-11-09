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

from flaggems_vllm.runtime.backend._ascend.FLA.chunk_gated_delta_rule_fwd_intra_npu import (
    chunk_gated_delta_rule_fwd_intra_npu_unfused,
    chunk_scaled_dot_kkt_fwd_950dt,
    recompute_w_u_fwd_950dt,
    solve_tril_950dt,
)
from flaggems_vllm.runtime.backend._ascend.FLA.chunk_gated_delta_rule_fwd_intra_npu_fused import (
    chunk_gated_delta_rule_fwd_intra_npu_fused,
)
from flaggems_vllm.runtime.backend._ascend.FLA.chunk_kda import chunk_kda
from flaggems_vllm.runtime.backend._ascend.FLA.chunk_kda_with_fused_gate import (
    chunk_kda_with_fused_gate,
)
from flaggems_vllm.runtime.backend._ascend.FLA.fused_recurrent_gated_delta_rule import (
    fused_recurrent_gated_delta_rule_fwd,
)
from flaggems_vllm.runtime.backend._ascend.FLA.fused_recurrent_gated_delta_rule_packed_decode import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from flaggems_vllm.runtime.backend._ascend.FLA.fused_recurrent_kda import (
    fused_recurrent_kda,
)
from flaggems_vllm.runtime.backend._ascend.FLA.gdn_chunk_recurrent_950dt import (
    chunk_gdn_950dt,
    fused_recurrent_gdn_950dt,
)

__all__ = [
    "chunk_gated_delta_rule_fwd_intra_npu_fused",
    "chunk_gated_delta_rule_fwd_intra_npu_unfused",
    "chunk_gdn_950dt",
    "chunk_kda",
    "chunk_kda_with_fused_gate",
    "chunk_scaled_dot_kkt_fwd_950dt",
    "fused_recurrent_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_recurrent_gdn_950dt",
    "fused_recurrent_kda",
    "recompute_w_u_fwd_950dt",
    "solve_tril_950dt",
]
