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
from flaggems_vllm.runtime.backend._ascend.ops.dequantize_and_gather_k_cache import (
    dequantize_and_gather_k_cache,
)
from flaggems_vllm.runtime.backend._ascend.ops.fused_add_rms_norm import (
    fused_add_rms_norm,
)
from flaggems_vllm.runtime.backend._ascend.ops.fused_inv_rope_int8_quant import (
    fused_inv_rope_int8_quant,
)
from flaggems_vllm.runtime.backend._ascend.ops.fused_moe import (
    fused_experts_impl,
    inplace_fused_experts,
    outplace_fused_experts,
)
from flaggems_vllm.runtime.backend._ascend.ops.gemma_rms_norm import gemma_rms_norm
from flaggems_vllm.runtime.backend._ascend.ops.group_list_cumsum import (
    group_list_cumsum,
)
from flaggems_vllm.runtime.backend._ascend.ops.grouped_topk import grouped_topk
from flaggems_vllm.runtime.backend._ascend.ops.hyperconnection import (
    qwen4_hc_inject_combine,
)
from flaggems_vllm.runtime.backend._ascend.ops.indexer_epilogue import indexer_epilogue
from flaggems_vllm.runtime.backend._ascend.ops.indexer_gemm_score import (
    indexer_gemm_score,
)
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
from flaggems_vllm.runtime.backend._ascend.ops.kpool_state_compress import (
    glm5_next_kpool_state_compress_and_write_cache_triton as kpool_state_compress,
)
from flaggems_vllm.runtime.backend._ascend.ops.kv_rmsnorm_rope_cache import (
    kv_rmsnorm_rope_cache,
)
from flaggems_vllm.runtime.backend._ascend.ops.lightning_indexer import (
    lightning_indexer,
)
from flaggems_vllm.runtime.backend._ascend.ops.mhc_pre import npu_mhc_pre
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
    "add_rms_norm",
    "build_compressor_metadata",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule_fwd",
    "combine_topk_swa_indices",
    "compressor",
    "compressor_prepared",
    "compute_slot_mapping_parallel",
    "dequantize_and_gather_k_cache",
    "fused_add_rms_norm",
    "fused_experts_impl",
    "fused_inv_rope_int8_quant",
    "gather_conv_state",
    "gather_kda_state",
    "gemma_rms_norm",
    "group_list_cumsum",
    "grouped_topk",
    "indexer_epilogue",
    "indexer_gemm_score",
    "inplace_fused_experts",
    "kda_gate_cumsum_triton",
    "kpool_state_compress",
    "kv_rmsnorm_rope_cache",
    "lightning_indexer",
    "npu_mhc_pre",
    "outplace_fused_experts",
    "pack_seq_triton",
    "paged_scatter_triton",
    "per_token_group_quant_fp8",
    "persistent_topk",
    "ple_state_scatter_",
    "prepare_compressor_workspace",
    "qwen4_compress_norm_mrope_store_groups",
    "qwen4_hc_inject_combine",
    "qwen4_qsa_mqa_paged_dot",
    "qwen4_store_qsa_kv_rows",
    "scaled_int8_quant",
    "scatter_conv_state",
    "scatter_kda_state",
    "sparse_attn_sharedkv",
    "SUPPORTED_FP8_DTYPE",
    "swiglu",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
    "topk_softplus_sqrt",
    "unpack_seq_triton",
]
