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

from flaggems_vllm.runtime.backend._hygon.ops.compress_norm_mrope import (  # noqa: F401
    qwen4_compress_norm_mrope_store_groups,
)
from flaggems_vllm.runtime.backend._hygon.ops.fused_moe import (  # noqa: F401
    fused_experts_impl,
    inplace_fused_experts,
    outplace_fused_experts,
)
from flaggems_vllm.runtime.backend._hygon.ops.hyperconnection import (
    qwen4_hc_inject_combine,
)
from flaggems_vllm.runtime.backend._hygon.ops.per_token_group_quant_fp8 import (
    SUPPORTED_FP8_DTYPE,
    per_token_group_quant_fp8,
)
from flaggems_vllm.runtime.backend._hygon.ops.ple_state import ple_state_scatter_
from flaggems_vllm.runtime.backend._hygon.ops.qsa import qwen4_store_qsa_kv_rows
from flaggems_vllm.runtime.backend._hygon.ops.qsa_mqa import qwen4_qsa_mqa_paged_dot
from flaggems_vllm.runtime.backend._hygon.ops.scaled_int8_quant import scaled_int8_quant
from flaggems_vllm.runtime.backend._hygon.ops.triton_scaled_mm import triton_scaled_mm

__all__ = [
    "SUPPORTED_FP8_DTYPE",
    "fused_experts_impl",
    "inplace_fused_experts",
    "outplace_fused_experts",
    "per_token_group_quant_fp8",
    "qwen4_store_qsa_kv_rows",
    "qwen4_hc_inject_combine",
    "ple_state_scatter_",
    "qwen4_qsa_mqa_paged_dot",
    "qwen4_compress_norm_mrope_store_groups",
    "scaled_int8_quant",
    "triton_scaled_mm",
]
