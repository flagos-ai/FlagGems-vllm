# isort: off
from flaggems_vllm.ops.add_rms_norm import add_rms_norm
from flaggems_vllm.ops.apply_repetition_penalties import apply_repetition_penalties
from flaggems_vllm.ops.bincount import bincount
from flaggems_vllm.ops.chunk_gated_delta_rule import chunk_gated_delta_rule
from flaggems_vllm.ops.concat_and_cache_mla import concat_and_cache_mla
from flaggems_vllm.ops.cp_gather_indexer_k_quant_cache import (
    cp_gather_indexer_k_quant_cache,
)
from flaggems_vllm.ops.cross_entropy_loss import cross_entropy_loss
from flaggems_vllm.ops.cutlass_scaled_mm import cutlass_scaled_mm
from flaggems_vllm.ops.deepseek_v4_attention_combine_topk_swa_indices import (
    combine_topk_swa_indices,
)
from flaggems_vllm.ops.deepseek_v4_attention_compute_global_topk_indices_and_lens import (
    compute_global_topk_indices_and_lens,
)
from flaggems_vllm.ops.deepseek_v4_attention_dequantize_and_gather_k_cache import (
    dequantize_and_gather_k_cache,
)
from flaggems_vllm.ops.deepseek_v4_attention_fused_q_kv_rmsnorm import (
    fused_q_kv_rmsnorm,
)
from flaggems_vllm.ops.DSA.bin_topk import bucket_sort_topk
from flaggems_vllm.ops.FLA import (
    chunk_gated_delta_rule_fwd,
    fused_recurrent_gated_delta_rule_fwd,
)
from flaggems_vllm.ops.attention import (
    flash_attention_forward,
    flash_attn_varlen_func,
    flash_attn_varlen_opt_func,
)
from flaggems_vllm.ops.flash_mla import flash_mla
from flaggems_vllm.ops.flash_mla_with_kvcache import flash_mla_with_kvcache
from flaggems_vllm.ops.flashmla_sparse import flash_mla_sparse_fwd
from flaggems_vllm.ops.fused_add_rms_norm import fused_add_rms_norm
from flaggems_vllm.ops.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert import (
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
)
from flaggems_vllm.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant
from flaggems_vllm.ops.fused_indexer_q_rope_quant import fused_indexer_q_rope_quant
from flaggems_vllm.ops.fused_moe import (
    dispatch_fused_moe_kernel,
    fused_experts_impl,
    inplace_fused_experts,
    invoke_fused_moe_triton_kernel,
    outplace_fused_experts,
)
from flaggems_vllm.ops.geglu import dgeglu, geglu
from flaggems_vllm.ops.gelu_and_mul import gelu_and_mul
from flaggems_vllm.ops.grouped_topk import grouped_topk
from flaggems_vllm.ops.indexer_k_quant_and_cache import indexer_k_quant_and_cache
from flaggems_vllm.ops.instance_norm import instance_norm
from flaggems_vllm.ops.mhc import (
    hc_head_fused_kernel,
    hc_head_fused_kernel_ref,
    mhc_bwd,
    mhc_bwd_ref,
    mhc_post,
    mhc_pre,
    sinkhorn_forward,
)
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size,
    moe_align_block_size_triton,
)
from flaggems_vllm.ops.moe_sum import moe_sum
from flaggems_vllm.ops.mul import mul, mul_
from flaggems_vllm.ops.mv import mv
from flaggems_vllm.ops.outer import outer
from flaggems_vllm.ops.pack_seq import pack_seq_triton
from flaggems_vllm.ops.FLA import (
    parallel_nsa,
    parallel_nsa_compression,
)
from flaggems_vllm.ops.per_token_group_quant_fp8 import (
    SUPPORTED_FP8_DTYPE,
    per_token_group_quant_fp8,
)
from flaggems_vllm.ops.reglu import dreglu, reglu
from flaggems_vllm.ops.reshape_and_cache import reshape_and_cache
from flaggems_vllm.ops.reshape_and_cache_flash import reshape_and_cache_flash
from flaggems_vllm.ops.rotary_embedding import apply_rotary_pos_emb
from flaggems_vllm.ops.rwkv_ka_fusion import rwkv_ka_fusion
from flaggems_vllm.ops.rwkv_mm_sparsity import rwkv_mm_sparsity
from flaggems_vllm.ops.silu_and_mul import silu_and_mul, silu_and_mul_out
from flaggems_vllm.ops.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp,
    silu_and_mul_with_clamp_out,
)
from flaggems_vllm.ops.skip_layernorm import skip_layer_norm
from flaggems_vllm.ops.sparse_attention import sparse_attn_triton
from flaggems_vllm.ops.stage_deepseek_v4_mega_moe_inputs import (
    stage_deepseek_v4_mega_moe_inputs,
)
from flaggems_vllm.ops.swiglu import dswiglu, swiglu
from flaggems_vllm.ops.top_k_per_row_decode import top_k_per_row_decode
from flaggems_vllm.ops.top_k_per_row_prefill import top_k_per_row_prefill
from flaggems_vllm.ops.topk_softmax import topk_softmax
from flaggems_vllm.ops.topk_softplus_sqrt import topk_softplus_sqrt
from flaggems_vllm.ops.unpack_seq import unpack_seq_triton
from flaggems_vllm.ops.weightnorm import (
    weight_norm_interface,
    weight_norm_interface_backward,
)
from flaggems_vllm.ops.weight_norm import weight_norm

# isort: on

__all__ = [
    "add_rms_norm",
    "apply_repetition_penalties",
    "apply_rotary_pos_emb",
    "bincount",
    "bucket_sort_topk",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd",
    "combine_topk_swa_indices",
    "compute_global_topk_indices_and_lens",
    "concat_and_cache_mla",
    "cp_gather_indexer_k_quant_cache",
    "cross_entropy_loss",
    "cutlass_scaled_mm",
    "dequantize_and_gather_k_cache",
    "dgeglu",
    "dispatch_fused_moe_kernel",
    "dreglu",
    "dswiglu",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flash_attn_varlen_opt_func",
    "flash_mla",
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
    "fused_add_rms_norm",
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
    "fused_experts_impl",
    "fused_indexer_q_rope_quant",
    "fused_inv_rope_fp8_quant",
    "fused_q_kv_rmsnorm",
    "fused_recurrent_gated_delta_rule_fwd",
    "geglu",
    "gelu_and_mul",
    "grouped_topk",
    "hc_head_fused_kernel",
    "hc_head_fused_kernel_ref",
    "indexer_k_quant_and_cache",
    "inplace_fused_experts",
    "instance_norm",
    "invoke_fused_moe_triton_kernel",
    "mhc_bwd",
    "mhc_bwd_ref",
    "mhc_post",
    "mhc_pre",
    "moe_align_block_size",
    "moe_align_block_size_triton",
    "moe_sum",
    "mul",
    "mul_",
    "mv",
    "outer",
    "outplace_fused_experts",
    "parallel_nsa",
    "parallel_nsa_compression",
    "pack_seq_triton",
    "per_token_group_quant_fp8",
    "reglu",
    "reshape_and_cache",
    "reshape_and_cache_flash",
    "rwkv_ka_fusion",
    "rwkv_mm_sparsity",
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
    "topk_softmax",
    "topk_softplus_sqrt",
    "unpack_seq_triton",
    "weight_norm",
    "weight_norm_interface",
    "weight_norm_interface_backward",
]
