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


from flaggems_vllm.runtime.backend._ascend.ops.add_rms_norm import add_rms_norm
from flaggems_vllm.runtime.backend._ascend.ops.causal_conv1d_fn import causal_conv1d_fn
from flaggems_vllm.runtime.backend._ascend.ops.causal_conv1d_update import (
    causal_conv1d_update,
)
from flaggems_vllm.runtime.backend._ascend.ops.chunk_gated_delta_rule_fwd import (
    chunk_gated_delta_rule_fwd,
)
from flaggems_vllm.runtime.backend._ascend.ops.compress_norm_mrope import (
    qwen4_compress_norm_mrope_store_groups,
)
from flaggems_vllm.runtime.backend._ascend.ops.compressor import (
    build_compressor_metadata,
    compressor,
    compressor_prepared,
    prepare_compressor_workspace,
)
from flaggems_vllm.runtime.backend._ascend.ops.deepseek_v4_attention_combine_topk_swa_indices import (
    combine_topk_swa_indices,
)
from flaggems_vllm.runtime.backend._ascend.ops.fused_moe import (
    fused_experts_impl,
    inplace_fused_experts,
    outplace_fused_experts,
)
from flaggems_vllm.runtime.backend._ascend.ops.grouped_topk import grouped_topk
from flaggems_vllm.runtime.backend._ascend.ops.hyperconnection import (
    qwen4_hc_inject_combine,
)
from flaggems_vllm.runtime.backend._ascend.ops.indexer_epilogue import indexer_epilogue
from flaggems_vllm.runtime.backend._ascend.ops.kda_conv_gather import gather_conv_state
from flaggems_vllm.runtime.backend._ascend.ops.kda_conv_scatter import (
    scatter_conv_state,
)
from flaggems_vllm.runtime.backend._ascend.ops.kda_gate_cumsum import (
    kda_gate_cumsum_triton,
)
from flaggems_vllm.runtime.backend._ascend.ops.kda_state_gather import gather_kda_state
from flaggems_vllm.runtime.backend._ascend.ops.kda_state_scatter import (
    scatter_kda_state,
)
from flaggems_vllm.runtime.backend._ascend.ops.pack_seq import pack_seq_triton
from flaggems_vllm.runtime.backend._ascend.ops.paged_scatter import paged_scatter_triton
from flaggems_vllm.runtime.backend._ascend.ops.per_token_group_quant_fp8 import (
    SUPPORTED_FP8_DTYPE,
    per_token_group_quant_fp8,
)
from flaggems_vllm.runtime.backend._ascend.ops.persistent_topk import persistent_topk
from flaggems_vllm.runtime.backend._ascend.ops.ple_state import ple_state_scatter_
from flaggems_vllm.runtime.backend._ascend.ops.qsa import qwen4_store_qsa_kv_rows
from flaggems_vllm.runtime.backend._ascend.ops.qsa_mqa import qwen4_qsa_mqa_paged_dot
from flaggems_vllm.runtime.backend._ascend.ops.scaled_int8_quant import (
    scaled_int8_quant,
)
from flaggems_vllm.runtime.backend._ascend.ops.slot_mapping import (
    compute_slot_mapping_parallel,
)
from flaggems_vllm.runtime.backend._ascend.ops.sparse_attn_sharedkv import (
    sparse_attn_sharedkv,
)
from flaggems_vllm.runtime.backend._ascend.ops.swiglu import swiglu
from flaggems_vllm.runtime.backend._ascend.ops.top_k_per_row_decode import (
    top_k_per_row_decode,
)
from flaggems_vllm.runtime.backend._ascend.ops.top_k_per_row_prefill import (
    top_k_per_row_prefill,
)
from flaggems_vllm.runtime.backend._ascend.ops.topk_softplus_sqrt import (
    topk_softplus_sqrt,
)
from flaggems_vllm.runtime.backend._ascend.ops.unpack_seq import unpack_seq_triton

__all__ = [
    "SUPPORTED_FP8_DTYPE",
    "add_rms_norm",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "fused_experts_impl",
    "grouped_topk",
    "inplace_fused_experts",
    "outplace_fused_experts",
    "qwen4_store_qsa_kv_rows",
    "qwen4_hc_inject_combine",
    "pack_seq_triton",
    "per_token_group_quant_fp8",
    "ple_state_scatter_",
    "qwen4_qsa_mqa_paged_dot",
    "qwen4_compress_norm_mrope_store_groups",
    "scaled_int8_quant",
    "sparse_attn_sharedkv",
    "swiglu",
    "chunk_gated_delta_rule_fwd",
    "persistent_topk",
    "compressor",
    "compressor_prepared",
    "prepare_compressor_workspace",
    "build_compressor_metadata",
    "combine_topk_swa_indices",
    "top_k_per_row_prefill",
    "top_k_per_row_decode",
    "topk_softplus_sqrt",
    "kda_gate_cumsum_triton",
    "gather_kda_state",
    "scatter_kda_state",
    "gather_conv_state",
    "scatter_conv_state",
    "paged_scatter_triton",
    "indexer_epilogue",
    "compute_slot_mapping_parallel",
    "unpack_seq_triton",
]
