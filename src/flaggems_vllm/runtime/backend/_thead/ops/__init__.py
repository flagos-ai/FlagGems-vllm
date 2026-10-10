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


from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
    fused_experts_impl,
    inplace_fused_experts,
    outplace_fused_experts,
)
from flaggems_vllm.runtime.backend._thead.ops.gemma_rms_norm import gemma_rms_norm
from flaggems_vllm.runtime.backend._thead.ops.int8_einsum import int8_einsum
from flaggems_vllm.runtime.backend._thead.ops.per_token_group_quant_fp8 import (
    SUPPORTED_FP8_DTYPE,
    per_token_group_quant_fp8,
)
from flaggems_vllm.runtime.backend._thead.ops.persistent_topk import persistent_topk
from flaggems_vllm.runtime.backend._thead.ops.topk_softplus_sqrt import (
    topk_softplus_sqrt,
)
from flaggems_vllm.runtime.backend._thead.ops.w8a8_block_int8_bmm import (
    w8a8_block_int8_bmm,
)

__all__ = [
    "SUPPORTED_FP8_DTYPE",
    "fused_experts_impl",
    "gemma_rms_norm",
    "inplace_fused_experts",
    "int8_einsum",
    "outplace_fused_experts",
    "SUPPORTED_FP8_DTYPE",
    "per_token_group_quant_fp8",
    "persistent_topk",
    "topk_softplus_sqrt",
    "w8a8_block_int8_bmm",
]
