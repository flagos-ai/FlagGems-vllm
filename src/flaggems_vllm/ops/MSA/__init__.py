"""Triton H100 v0 — Paged KV Cache version.

Uses vLLM's paged KV cache format:
  kv_cache: [num_blocks, num_kv_heads, 128, 2*head_dim]  K=[..., :head_dim] V=[..., head_dim:]
  index_kv_cache: [num_blocks, 128, head_dim]
  block_table: [batch, max_blocks]
"""

from .index_topk import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_decode_score,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from .sparse_attn import minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode

__all__ = [
    "SPARSE_BLOCK_SIZE",
    "minimax_m3_index_decode",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
]
