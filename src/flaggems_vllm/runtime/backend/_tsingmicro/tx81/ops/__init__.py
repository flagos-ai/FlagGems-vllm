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

from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.attention import (
    flash_attention_forward,
    flash_attn_varlen_func,
)

# tx81 specialized vLLM operators, migrated from FlagGems
# runtime/backend/_tsingmicro/tx81/fused/. Functions re-exported here override
# the generic implementations through SpecOpRegistrar (matched by function
# name), so every public entry must keep the generic-op signature.
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.fla import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_fwd,
)
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.flash_mla import flash_mla
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.flashmla_sparse import (
    flash_mla_sparse_fwd,
)
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.fused_add_rms_norm import (
    fused_add_rms_norm,
)
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.fused_moe import (
    dispatch_fused_moe_kernel,
    fused_experts_impl,
    inplace_fused_experts,
    invoke_fused_moe_triton_kernel,
    outplace_fused_experts,
)
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.grouped_topk import grouped_topk
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.mhc_post import mhc_post
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.mhc_pre import mhc_pre
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.moe_align_block_size import (
    moe_align_block_size,
    moe_align_block_size_triton,
)
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.reshape_and_cache_flash import (
    reshape_and_cache_flash,
)
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.silu_and_mul import (
    silu_and_mul,
    silu_and_mul_out,
)
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp,
    silu_and_mul_with_clamp_out,
)

__all__ = [
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flash_mla",
    "flash_mla_sparse_fwd",
    "dispatch_fused_moe_kernel",
    "fused_experts_impl",
    "inplace_fused_experts",
    "invoke_fused_moe_triton_kernel",
    "outplace_fused_experts",
    "moe_align_block_size",
    "moe_align_block_size_triton",
    "mhc_post",
    "mhc_pre",
    "fused_add_rms_norm",
    "grouped_topk",
    "reshape_and_cache_flash",
    "silu_and_mul",
    "silu_and_mul_out",
    "silu_and_mul_with_clamp",
    "silu_and_mul_with_clamp_out",
]
