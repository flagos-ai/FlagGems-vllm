# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from flaggems_vllm.ops.FLA.bwd_preprocess import parallel_attn_bwd_preprocess
from flaggems_vllm.ops.FLA.chunk import chunk_gated_delta_rule_fwd
from flaggems_vllm.ops.FLA.chunk_gdn2 import chunk_gdn2
from flaggems_vllm.ops.FLA.chunk_kda import chunk_kda
from flaggems_vllm.ops.FLA.fused_recurrent_gated_delta_rule import (
    fused_recurrent_gated_delta_rule_fwd,
)
from flaggems_vllm.ops.FLA.fused_recurrent_gated_delta_rule_packed_decode import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from flaggems_vllm.ops.FLA.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_token_indices,
)
from flaggems_vllm.ops.FLA.parallel_nsa import parallel_nsa
from flaggems_vllm.ops.FLA.parallel_nsa_compression import parallel_nsa_compression
from flaggems_vllm.ops.FLA.triton_ops_helper import autotune_cache_kwargs, exp, log

__all__ = [
    "autotune_cache_kwargs",
    "chunk_gated_delta_rule_fwd",
    "exp",
    "chunk_gdn2",
    "chunk_kda",
    "fused_recurrent_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "log",
    "parallel_attn_bwd_preprocess",
    "parallel_nsa",
    "parallel_nsa_compression",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "prepare_token_indices",
]

# Select backend implementations before ops and the top-level package import
# these bindings. Backend FLA packages are not scanned by the ops registrar.
from flaggems_vllm import runtime as _runtime

if _runtime.device.vendor_name == "ascend":
    from flaggems_vllm.runtime.backend._ascend.FLA import (  # noqa: F811
        fused_recurrent_gated_delta_rule_fwd,
        fused_recurrent_gated_delta_rule_packed_decode,
    )
