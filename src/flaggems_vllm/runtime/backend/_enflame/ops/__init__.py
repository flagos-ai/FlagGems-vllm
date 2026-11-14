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

"""Enflame (GCU) specialized operators.

Seeded as verbatim copies of the device-agnostic implementations in
``flaggems_vllm.ops`` so the backend owns an explicit, adaptable copy of
the full operator surface. ``scaled_int8_quant`` keeps its GCU-specific
implementation. Copied operators keep the exact generic behavior until
they are individually adapted for GCU.

Intra-op cross imports inside the copied modules still resolve to the
generic ``flaggems_vllm.ops`` package; rewire them when adapting an
operator locally.
"""

# Keep vllm.third_party.deep_gemm importable on hosts whose torch build lacks
# the CUDA shared libraries its vendored _C extension needs (test/benchmark
# modules import deep_gemm.utils.math at module level). See the module
# docstring of _vllm_deep_gemm_import_compat for the exact mechanism.
import sys

import torch

from flaggems_vllm.runtime import device as _runtime_device
from flaggems_vllm.runtime.backend._enflame.ops._vllm_deep_gemm_import_compat import (
    install_vllm_deep_gemm_import_compat,
)

install_vllm_deep_gemm_import_compat()  # 库兼容 shim(非 GCUIR),保留

# vLLM accuracy tests call get_paged_mqa_logits_metadata with torch's
# multi_processor_count, which the GCU deep_gemm C++ API rejects by
# hard-exiting the process (killing the whole pytest session before the
# op under test runs); its topsaten path likewise rejects the standard
# vLLM 2D tensor ranks. See the module docstring of
# _vllm_paged_mqa_logits_compat for the exact mechanism.
from flaggems_vllm.runtime.backend._enflame.ops._vllm_paged_mqa_logits_compat import (
    install_vllm_paged_mqa_logits_compat,
)

install_vllm_paged_mqa_logits_compat()  # 库兼容 shim(非 GCUIR),保留

# Make vLLM's own fused_experts_impl reference runnable on GCU (tests compare
# FlagGems fused_moe against it): fall back to a GCU device name when vLLM
# cannot detect a platform, and provide the missing _C/_moe_C ops through
# FlagGems Triton implementations. No-op wherever vLLM works natively.
from flaggems_vllm.runtime.backend._enflame.ops._vllm_moe_ref_compat import (
    install_vllm_moe_ref_compat,
)

install_vllm_moe_ref_compat()  # 库兼容 shim(非 GCUIR),保留

# flaggems_vllm 的 LibEntry(不同于 LibTuner)未暴露上游测试契约的 cache_key
# (tests/test_tle_capabilities.py 断言 @libentry() 内核可访问该属性);纯增量
# 类属性兼容补丁,仅在该属性缺失时生效,委托到 LibEntry 已解析的 jit_function。
from flaggems_vllm.runtime.backend._enflame.ops._flaggems_libentry_compat import (
    install_flaggems_libentry_compat,
)

install_flaggems_libentry_compat()  # 库兼容 shim(非 GCUIR),保留

# isort: off
from flaggems_vllm.runtime.backend._enflame.ops.act_quant import act_quant_triton
from flaggems_vllm.runtime.backend._enflame.ops.add_rms_norm import add_rms_norm
from flaggems_vllm.runtime.backend._enflame.ops.apply_repetition_penalties import (
    apply_repetition_penalties,
)
from flaggems_vllm.runtime.backend._enflame.ops.attn_res import attn_res
from flaggems_vllm.runtime.backend._enflame.ops.beam_search_score import (
    beam_search_score,
    beam_search_score_,
)
from flaggems_vllm.runtime.backend._enflame.ops.bf16_paged_mqa_logits import (
    bf16_paged_mqa_logits,
)
from flaggems_vllm.runtime.backend._enflame.ops.bincount import bincount
from flaggems_vllm.runtime.backend._enflame.ops.chunk_gated_delta_rule import (
    chunk_gated_delta_rule,
)
from flaggems_vllm.runtime.backend._enflame.ops.concat_and_cache_mla import (
    concat_and_cache_mla,
)
from flaggems_vllm.runtime.backend._enflame.ops.concat_mla_q import concat_mla_q
from flaggems_vllm.runtime.backend._enflame.ops.cp_gather_indexer_k_quant_cache import (
    cp_gather_indexer_k_quant_cache,
)
from flaggems_vllm.runtime.backend._enflame.ops.cross_entropy_loss import (
    cross_entropy_loss,
)
from flaggems_vllm.runtime.backend._enflame.ops.cutlass_scaled_mm import (
    cutlass_scaled_mm,
)
from flaggems_vllm.runtime.backend._enflame.ops.deepseek_v4_attention_combine_topk_swa_indices import (
    combine_topk_swa_indices,
)
from flaggems_vllm.runtime.backend._enflame.ops.deepseek_v4_attention_compute_global_topk_indices_and_lens import (
    compute_global_topk_indices_and_lens,
)
from flaggems_vllm.runtime.backend._enflame.ops.deepseek_v4_attention_dequantize_and_gather_k_cache import (
    dequantize_and_gather_k_cache,
)
from flaggems_vllm.runtime.backend._enflame.ops.deepseek_v4_attention_fused_q_kv_rmsnorm import (
    fused_q_kv_rmsnorm,
)
from flaggems_vllm.runtime.backend._enflame.ops.DSA.bin_topk import bucket_sort_topk
from flaggems_vllm.runtime.backend._enflame.ops.DSA.indexer_k_tiled import (
    triton_lighting_indexer_k_tiled_interface,
)
from flaggems_vllm.runtime.backend._enflame.ops.DSA.sparse_mla import (
    triton_sparse_mla_fwd_interface,
)
from flaggems_vllm.runtime.backend._enflame.ops.FLA import (
    chunk_kda_scaled_dot_kkt_fwd,
    chunk_gated_delta_rule_fwd,
    chunk_gla,
    chunk_kda,
    fused_recurrent_gated_delta_rule_fwd,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_kda,
)
from flaggems_vllm.runtime.backend._enflame.ops.gdn_w8a16 import gdn_w8a16
from flaggems_vllm.runtime.backend._enflame.ops.attention import (
    flash_attention_forward,
    flash_attn_varlen_func,
    flash_attn_varlen_opt_func,
)
from flaggems_vllm.runtime.backend._enflame.ops.flash_attn_varlen_func_w8a8_fp8 import (
    flash_attn_varlen_func_w8a8_fp8,
)
from flaggems_vllm.runtime.backend._enflame.ops.flash_mla import flash_mla
from flaggems_vllm.runtime.backend._enflame.ops.flash_mla_with_kvcache_w8a8 import (
    flash_mla_with_kvcache_w8a8,
)
from flaggems_vllm.runtime.backend._enflame.ops.flash_mla_with_kvcache import (
    flash_mla_with_kvcache,
)
from flaggems_vllm.runtime.backend._enflame.ops.flashmla_sparse import (
    flash_mla_sparse_fwd,
)
from flaggems_vllm.runtime.backend._enflame.ops.expand_pools_and_append_tail import (
    expand_pools_and_append_tail,
)
from flaggems_vllm.runtime.backend._enflame.ops.fp8_einsum import fp8_einsum

from flaggems_vllm.runtime.backend._enflame.ops.fp8_fp4_mqa_logits import (
    fp8_fp4_mqa_logits,
)
from flaggems_vllm.runtime.backend._enflame.ops.fp8_fp4_paged_mqa_logits import (
    fp8_fp4_paged_mqa_logits,
)
from flaggems_vllm.runtime.backend._enflame.ops.fp8_fp4_paged_mqa_logits_graph import (
    fp8_fp4_paged_mqa_logits_graph,
)

# Backend-local E4M3FN codecs; importing the module installs the GCU
# implementation into the generic fp8_storage entrypoints (no public host op).
from flaggems_vllm.runtime.backend._enflame.ops import fp8_storage  # noqa: F401
from flaggems_vllm.runtime.backend._enflame.ops.fused_add_rms_norm import (
    fused_add_rms_norm,
)
from flaggems_vllm.runtime.backend._enflame.ops.fused_allreduce_rms_norm import (
    fused_allreduce_rms_norm,
)
from flaggems_vllm.runtime.backend._enflame.ops.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert import (
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
)
from flaggems_vllm.runtime.backend._enflame.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)

# Make vLLM's own fused_inv_rope_fp8_quant reference runnable on GCU
# (benchmarks time it as the baseline): its kernel's int64 positions scalar
# load aborts the device (triton_i64_narrow_load); the shim re-exposes
# positions as compact int32 words (zero-copy) before delegating. No-op
# wherever vLLM works natively.
from flaggems_vllm.runtime.backend._enflame.ops._vllm_inv_rope_ref_compat import (
    install_vllm_inv_rope_ref_compat,
)

install_vllm_inv_rope_ref_compat()  # 库兼容 shim(非 GCUIR),保留

from flaggems_vllm.runtime.backend._enflame.ops.fused_indexer_q_rope_quant import (
    fused_indexer_q_rope_quant,
)
from flaggems_vllm.runtime.backend._enflame.ops.fused_marlin_moe import fused_marlin_moe
from flaggems_vllm.runtime.backend._enflame.ops.fused_moe import (
    dispatch_fused_moe_kernel,
    fused_experts_impl,
    inplace_fused_experts,
    invoke_fused_moe_triton_kernel,
    outplace_fused_experts,
)
from flaggems_vllm.runtime.backend._enflame.ops.geglu import dgeglu, geglu
from flaggems_vllm.runtime.backend._enflame.ops.gelu_and_mul import gelu_and_mul
from flaggems_vllm.runtime.backend._enflame.ops.gemma_rms_norm import gemma_rms_norm
from flaggems_vllm.runtime.backend._enflame.ops.grouped_topk import grouped_topk
from flaggems_vllm.runtime.backend._enflame.ops.indexer_k_quant_and_cache import (
    indexer_k_quant_and_cache,
)
from flaggems_vllm.runtime.backend._enflame.ops.instance_norm import instance_norm
from flaggems_vllm.runtime.backend._enflame.ops.ihc import (
    ihc_post,
    ihc_post_ref,
    ihc_pre,
    ihc_pre_ref,
)
from flaggems_vllm.runtime.backend._enflame.ops.mhc import (
    MixLayout,
    hc_head_fused_kernel,
    hc_head_fused_kernel_ref,
    mhc_bwd,
    mhc_bwd_ref,
    mhc_post,
    mhc_pre,
    sinkhorn_forward,
)
from flaggems_vllm.runtime.backend._enflame.ops.qwen4 import (
    ple_short_conv_,
    ple_gate_norm,
    ple_gate_norm_,
    ple_prefill_short_conv_,
    qsa_compress_groups_with_ratio,
    qsa_compress_norm_mrope_store_groups,
    qsa_mqa_paged,
    qsa_sparse_paged_attention,
    qsa_sparse_split_count,
    qsa_store_cache_rows,
    qsa_store_kv_cache_rows,
    qwen4_hc_combine_norm,
    ple_state_gather,
    ple_state_scatter_,
    qwen4_compress_norm_mrope_store_groups,
    qwen4_grouped_gemma_rmsnorm,
    qwen4_hc_gate_reduce,
    qwen4_hc_inject_combine,
    qwen4_qsa_mqa_paged_dot,
    qwen4_store_qsa_kv_rows,
    qwen4_vendor_compress_qsa_groups,
    qwen4_vendor_qsa_mqa_paged,
    qwen4_vendor_store_qsa_rows,
    qsa_prepare,
)
from flaggems_vllm.runtime.backend._enflame.ops.minimax_m3_sparse_attn import (
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from flaggems_vllm.runtime.backend._enflame.ops.moe_align_block_size import (
    moe_align_block_size,
    moe_align_block_size_no_tle,
    moe_align_block_size_triton,
)
from flaggems_vllm.runtime.backend._enflame.ops.moe_sum import moe_sum
from flaggems_vllm.runtime.backend._enflame.ops.mrope import mrope
from flaggems_vllm.runtime.backend._enflame.ops.mul import mul, mul_
from flaggems_vllm.runtime.backend._enflame.ops.mv import mv
from flaggems_vllm.runtime.backend._enflame.ops.outer import outer
from flaggems_vllm.runtime.backend._enflame.ops.pack_seq import pack_seq_triton
from flaggems_vllm.runtime.backend._enflame.ops.FLA import (
    parallel_nsa,
    parallel_nsa_compression,
)
from flaggems_vllm.runtime.backend._enflame.ops.per_token_group_quant_fp8 import (
    SUPPORTED_FP8_DTYPE,
    per_token_group_quant_fp8,
)
from flaggems_vllm.runtime.backend._enflame.ops.permute_copy import permute_copy
from flaggems_vllm.runtime.backend._enflame.ops.persistent_topk import persistent_topk
from flaggems_vllm.runtime.backend._enflame.ops.reglu import dreglu, reglu
from flaggems_vllm.runtime.backend._enflame.ops.reshape_and_cache import (
    reshape_and_cache,
)
from flaggems_vllm.runtime.backend._enflame.ops.reshape_and_cache_flash import (
    reshape_and_cache_flash,
)
from flaggems_vllm.runtime.backend._enflame.ops.rotary_embedding import (
    apply_rotary_pos_emb,
)
from flaggems_vllm.runtime.backend._enflame.ops.router_gemm import router_gemm
from flaggems_vllm.runtime.backend._enflame.ops.rwkv_ka_fusion import rwkv_ka_fusion
from flaggems_vllm.runtime.backend._enflame.ops.rwkv_mm_sparsity import rwkv_mm_sparsity
from flaggems_vllm.runtime.backend._enflame.ops.scaled_int8_quant import (
    scaled_int8_quant,
)
from flaggems_vllm.runtime.backend._enflame.ops.silu_and_mul import (
    silu_and_mul,
    silu_and_mul_out,
)
from flaggems_vllm.runtime.backend._enflame.ops.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp,
    silu_and_mul_with_clamp_out,
)
from flaggems_vllm.runtime.backend._enflame.ops.skip_layernorm import skip_layer_norm
from flaggems_vllm.runtime.backend._enflame.ops.sparse_attention import (
    sparse_attn_triton,
)
from flaggems_vllm.runtime.backend._enflame.ops.stage_deepseek_v4_mega_moe_inputs import (
    stage_deepseek_v4_mega_moe_inputs,
)
from flaggems_vllm.runtime.backend._enflame.ops.swiglu import dswiglu, swiglu
from flaggems_vllm.runtime.backend._enflame.ops.top_k_per_row_decode import (
    top_k_per_row_decode,
)
from flaggems_vllm.runtime.backend._enflame.ops.top_k_per_row_prefill import (
    top_k_per_row_prefill,
)
from flaggems_vllm.runtime.backend._enflame.ops.topk_softmax import topk_softmax
from flaggems_vllm.runtime.backend._enflame.ops.topk_softplus_sqrt import (
    topk_softplus_sqrt,
)
from flaggems_vllm.runtime.backend._enflame.ops.triton_scaled_mm import triton_scaled_mm
from flaggems_vllm.runtime.backend._enflame.ops.triton_unified_attention import (
    triton_unified_attention,
)
from flaggems_vllm.runtime.backend._enflame.ops.unpack_seq import unpack_seq_triton
from flaggems_vllm.runtime.backend._enflame.ops.weightnorm import (
    weight_norm_interface,
    weight_norm_interface_backward,
)
from flaggems_vllm.runtime.backend._enflame.ops.weight_norm import weight_norm

# isort: on

from flaggems_vllm.runtime.backend._enflame.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from flaggems_vllm.runtime.backend._enflame.ops.data_movement import (
    concat_query,
    contiguous_copy,
    copy_to,
    gather_rows,
    gather_state_rows,
    pad_attention_heads,
    scatter_decode_tokens,
    scatter_state_rows,
    zero_state_rows,
)
from flaggems_vllm.runtime.backend._enflame.ops.fp32_router_gemm import fp32_router_gemm
from flaggems_vllm.runtime.backend._enflame.ops.fused_minimax_m3_qknorm_rope_kv_insert import (
    fused_minimax_m3_qknorm_rope_kv_insert,
)
from flaggems_vllm.runtime.backend._enflame.ops.kda_bounded import (
    chunk_kda_with_safe_gate,
    recurrent_kda,
    safe_kda_gate,
    safe_kda_gate_chunk_cumsum,
)
from flaggems_vllm.runtime.backend._enflame.ops.kpool import (
    fwht128_quant_fp8,
    hadamard128,
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
    kpool_seed_tail_cache,
)
from flaggems_vllm.runtime.backend._enflame.ops.mhc_with_norm import (
    mhc_fused_post_pre_with_norm,
    mhc_pre_with_norm,
    mhc_rms_norm,
)
from flaggems_vllm.runtime.backend._enflame.ops.paged_mqa import paged_mqa_logits
from flaggems_vllm.runtime.backend._enflame.ops.topk_sigmoid import topk_sigmoid
from flaggems_vllm.runtime.backend._enflame.ops.vit_attention import (
    vision_flash_attn_varlen,
    vit_flash_attn_wrapper,
)
from flaggems_vllm.runtime.backend._enflame.ops.w8a8_block_fp8_matmul import (
    w8a8_block_fp8_matmul,
)

# GCU Triton casts and stores float8_e4m3fn bit-exactly (verified on L600;
# see compiler_issues/triton_i64_narrow_load.md), but the generic
# kernel_supports_fp8_e4m3() vendor whitelist predates that verification and
# does not know enflame.  runtime.replace_customized_ops() only replaces
# functions, so the package-level SUPPORTED_FP8_DTYPE gate that
# tests/benchmarks read resolves to float32 on GCU unless this backend
# corrects it at import time.  fp8-quant ops in this package normalize int64
# index tensors to int32 on the host (triton_i64_narrow_load), so declaring
# fp8 support here is safe for every op consuming the gate.
if (
    _runtime_device.vendor_name == "enflame"
    and getattr(torch, "float8_e4m3fn", None) is not None
):
    _flaggems_pkg = sys.modules.get("flaggems_vllm")
    if (
        _flaggems_pkg is not None
        and getattr(_flaggems_pkg, "SUPPORTED_FP8_DTYPE", None) is torch.float32
    ):
        _flaggems_pkg.SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn

__all__ = [
    "act_quant_triton",
    "add_rms_norm",
    "apply_repetition_penalties",
    "apply_rotary_pos_emb",
    "attn_res",
    "beam_search_score",
    "beam_search_score_",
    "bf16_paged_mqa_logits",
    "bincount",
    "bucket_sort_topk",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd",
    "chunk_gla",
    "chunk_kda",
    "chunk_kda_scaled_dot_kkt_fwd",
    "chunk_kda_with_safe_gate",
    "combine_topk_swa_indices",
    "compute_global_topk_indices_and_lens",
    "concat_and_cache_mla",
    "concat_mla_q",
    "concat_query",
    "contiguous_copy",
    "copy_to",
    "cp_gather_indexer_k_quant_cache",
    "cross_entropy_loss",
    "cutlass_scaled_mm",
    "dequantize_and_gather_k_cache",
    "dgeglu",
    "dispatch_fused_moe_kernel",
    "dreglu",
    "dswiglu",
    "expand_pools_and_append_tail",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flash_attn_varlen_func_w8a8_fp8",
    "flash_attn_varlen_opt_func",
    "flash_mla",
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
    "flash_mla_with_kvcache_w8a8",
    "fp32_router_gemm",
    "fp8_einsum",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_paged_mqa_logits",
    "fp8_fp4_paged_mqa_logits_graph",
    "fused_add_rms_norm",
    "fused_allreduce_rms_norm",
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
    "fused_experts_impl",
    "fused_indexer_q_rope_quant",
    "fused_inv_rope_fp8_quant",
    "fused_marlin_moe",
    "fused_minimax_m3_qknorm_rope_kv_insert",
    "fused_q_kv_rmsnorm",
    "fused_recurrent_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_recurrent_kda",
    "fwht128_quant_fp8",
    "gather_rows",
    "gather_state_rows",
    "gdn_w8a16",
    "geglu",
    "gelu_and_mul",
    "gemma_rms_norm",
    "grouped_topk",
    "hadamard128",
    "hc_head_fused_kernel",
    "hc_head_fused_kernel_ref",
    "ihc_post",
    "ihc_post_ref",
    "ihc_pre",
    "ihc_pre_ref",
    "indexer_k_quant_and_cache",
    "inplace_fused_experts",
    "instance_norm",
    "invoke_fused_moe_triton_kernel",
    "kpool_compress_and_write_cache",
    "kpool_decode_update_and_maybe_write_cache_batched",
    "kpool_seed_tail_cache",
    "mhc_bwd",
    "mhc_bwd_ref",
    "mhc_fused_post_pre_with_norm",
    "mhc_post",
    "mhc_pre",
    "mhc_pre_with_norm",
    "mhc_rms_norm",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
    "MixLayout",
    "moe_align_block_size",
    "moe_align_block_size_no_tle",
    "moe_align_block_size_triton",
    "moe_sum",
    "mrope",
    "mul",
    "mul_",
    "mv",
    "outer",
    "outplace_fused_experts",
    "pack_seq_triton",
    "pad_attention_heads",
    "paged_mqa_logits",
    "parallel_nsa",
    "parallel_nsa_compression",
    "per_token_group_quant_fp8",
    "permute_copy",
    "persistent_topk",
    "ple_gate_norm",
    "ple_gate_norm_",
    "ple_prefill_short_conv_",
    "ple_short_conv_",
    "ple_state_gather",
    "ple_state_scatter_",
    "qsa_compress_groups_with_ratio",
    "qsa_compress_norm_mrope_store_groups",
    "qsa_mqa_paged",
    "qsa_prepare",
    "qsa_sparse_paged_attention",
    "qsa_sparse_split_count",
    "qsa_store_cache_rows",
    "qsa_store_kv_cache_rows",
    "qwen4_compress_norm_mrope_store_groups",
    "qwen4_grouped_gemma_rmsnorm",
    "qwen4_hc_combine_norm",
    "qwen4_hc_gate_reduce",
    "qwen4_hc_inject_combine",
    "qwen4_qsa_mqa_paged_dot",
    "qwen4_store_qsa_kv_rows",
    "qwen4_vendor_compress_qsa_groups",
    "qwen4_vendor_qsa_mqa_paged",
    "qwen4_vendor_store_qsa_rows",
    "recurrent_kda",
    "reglu",
    "reshape_and_cache",
    "reshape_and_cache_flash",
    "router_gemm",
    "rwkv_ka_fusion",
    "rwkv_mm_sparsity",
    "safe_kda_gate",
    "safe_kda_gate_chunk_cumsum",
    "scaled_int8_quant",
    "scatter_decode_tokens",
    "scatter_state_rows",
    "silu_and_mul",
    "silu_and_mul_out",
    "silu_and_mul_with_clamp",
    "silu_and_mul_with_clamp_out",
    "sinkhorn_forward",
    "skip_layer_norm",
    "sparse_attn_triton",
    "stage_deepseek_v4_mega_moe_inputs",
    "SUPPORTED_FP8_DTYPE",
    "swiglu",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
    "topk_sigmoid",
    "topk_softmax",
    "topk_softplus_sqrt",
    "triton_lighting_indexer_k_tiled_interface",
    "triton_scaled_mm",
    "triton_sparse_mla_fwd_interface",
    "triton_unified_attention",
    "unpack_seq_triton",
    "vision_flash_attn_varlen",
    "vit_flash_attn_wrapper",
    "w8a8_block_fp8_matmul",
    "weight_norm",
    "weight_norm_interface",
    "weight_norm_interface_backward",
    "zero_state_rows",
]
