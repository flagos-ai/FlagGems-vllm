# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Identity re-exports of the shared, public Core INT8 primitives."""

from flag_gems.fused.int8_sparse_indexer import (
    int8_indexer_cache_store,
    int8_indexer_paged_logits,
    int8_indexer_quantize,
)

__all__ = [
    "int8_indexer_cache_store",
    "int8_indexer_paged_logits",
    "int8_indexer_quantize",
]
