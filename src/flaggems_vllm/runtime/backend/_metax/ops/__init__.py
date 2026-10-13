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

from flaggems_vllm.runtime.backend._metax.ops.compress_norm_mrope import (
    qwen4_compress_norm_mrope_store_groups,
)
from flaggems_vllm.runtime.backend._metax.ops.fused_moe import (
    fused_experts_impl,
    inplace_fused_experts,
    outplace_fused_experts,
)
from flaggems_vllm.runtime.backend._metax.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from flaggems_vllm.runtime.backend._metax.ops.hyperconnection import (
    qwen4_hc_inject_combine,
)
from flaggems_vllm.runtime.backend._metax.ops.int8_einsum import int8_einsum
from flaggems_vllm.runtime.backend._metax.ops.per_token_group_quant_fp8 import (
    SUPPORTED_FP8_DTYPE,
    per_token_group_quant_fp8,
)
from flaggems_vllm.runtime.backend._metax.ops.ple_state import ple_state_scatter_
from flaggems_vllm.runtime.backend._metax.ops.qsa import qwen4_store_qsa_kv_rows
from flaggems_vllm.runtime.backend._metax.ops.qsa_mqa import qwen4_qsa_mqa_paged_dot
from flaggems_vllm.runtime.backend._metax.ops.scaled_int8_quant import scaled_int8_quant
from flaggems_vllm.runtime.backend._metax.ops.w8a8_block_int8_bmm import (
    w8a8_block_int8_bmm,
)
from flaggems_vllm.utils import triton_version_utils

# A missing optional TLE extension must not disable the independent INT8 ops.
if triton_version_utils.has_triton_tle():
    from flaggems_vllm.runtime.backend._metax.ops.persistent_topk import persistent_topk
else:
    from flaggems_vllm.ops.persistent_topk import persistent_topk

__all__ = [
    "int8_einsum",
    "w8a8_block_int8_bmm",
    "SUPPORTED_FP8_DTYPE",
    "per_token_group_quant_fp8",
    "persistent_topk",
    "qwen4_store_qsa_kv_rows",
    "qwen4_hc_inject_combine",
    "ple_state_scatter_",
    "qwen4_qsa_mqa_paged_dot",
    "qwen4_compress_norm_mrope_store_groups",
    "scaled_int8_quant",
    "fused_experts_impl",
    "inplace_fused_experts",
    "outplace_fused_experts",
    "fused_recurrent_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_packed_decode",
]
