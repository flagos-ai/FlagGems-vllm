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

from flaggems_vllm.ops.qwen4.hyperconnection import (
    can_use_hc_combine_norm_triton,
    can_use_hc_inject_triton,
    can_use_hc_triton,
    qwen4_grouped_gemma_rmsnorm,
    qwen4_hc_combine_norm,
    qwen4_hc_gate_reduce,
    qwen4_hc_inject_combine,
)
from flaggems_vllm.ops.qwen4.ple_fusion import (
    can_use_ple_gate_norm_triton,
    can_use_ple_prefill_conv_triton,
    ple_gate_norm,
    ple_gate_norm_,
    ple_prefill_short_conv_,
)
from flaggems_vllm.ops.qwen4.ple_state import ple_state_gather, ple_state_scatter_
from flaggems_vllm.ops.qwen4.qsa import (
    qsa_compress_groups_with_ratio,
    qsa_compress_norm_mrope_store_groups,
    qsa_mqa_paged,
    qsa_store_cache_rows,
    qsa_store_kv_cache_rows,
    qwen4_compress_norm_mrope_store_groups,
    qwen4_qsa_mqa_paged_dot,
    qwen4_store_qsa_kv_rows,
    qwen4_vendor_compress_qsa_groups,
    qwen4_vendor_qsa_mqa_paged,
    qwen4_vendor_store_qsa_rows,
)
from flaggems_vllm.ops.qwen4.qsa_attention import (
    qsa_sparse_paged_attention,
    qsa_sparse_split_count,
)

__all__ = [
    "qwen4_hc_combine_norm",
    "ple_gate_norm_",
    "ple_gate_norm",
    "ple_prefill_short_conv_",
    "qsa_mqa_paged",
    "qsa_store_cache_rows",
    "qsa_store_kv_cache_rows",
    "qsa_compress_groups_with_ratio",
    "qsa_compress_norm_mrope_store_groups",
    "qsa_sparse_paged_attention",
    "qsa_sparse_split_count",
    "can_use_hc_combine_norm_triton",
    "can_use_ple_gate_norm_triton",
    "can_use_ple_prefill_conv_triton",
    "can_use_hc_inject_triton",
    "can_use_hc_triton",
    "ple_state_gather",
    "ple_state_scatter_",
    "qwen4_compress_norm_mrope_store_groups",
    "qwen4_grouped_gemma_rmsnorm",
    "qwen4_hc_gate_reduce",
    "qwen4_hc_inject_combine",
    "qwen4_qsa_mqa_paged_dot",
    "qwen4_store_qsa_kv_rows",
    "qwen4_vendor_compress_qsa_groups",
    "qwen4_vendor_qsa_mqa_paged",
    "qwen4_vendor_store_qsa_rows",
]
