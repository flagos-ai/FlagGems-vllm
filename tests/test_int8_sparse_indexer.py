# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public identity edges; mathematics is tested in the shared Core package."""

import pytest


@pytest.mark.parametrize(
    "name",
    ["int8_indexer_quantize", "int8_indexer_cache_store", "int8_indexer_paged_logits"],
)
def test_public_int8_apis_are_the_shared_implementation(name):
    from flag_gems.fused import int8_sparse_indexer

    import flaggems_vllm

    assert getattr(flaggems_vllm, name) is getattr(int8_sparse_indexer, name)
