# SPDX-License-Identifier: Apache-2.0
"""MetaX FLA operator implementations."""

from flaggems_vllm.runtime.backend._metax.FLA.chunk_kda_with_fused_gate import (
    chunk_kda_with_fused_gate,
)

__all__ = [
    "chunk_kda_with_fused_gate",
]
