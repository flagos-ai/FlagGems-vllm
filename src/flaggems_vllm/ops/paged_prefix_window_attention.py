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

"""Paged decode over a disjoint prefix and a causal sliding tail."""

from flaggems_vllm import runtime


def is_paged_prefix_window_attention_available():
    """Check the vendor API before selecting this implementation."""
    if runtime.device.vendor_name != "metax":
        return False
    from flaggems_vllm.runtime.backend._metax.ops.paged_prefix_window_attention import (
        is_paged_prefix_window_attention_available as available,
    )

    return available()


def paged_prefix_window_attention(
    q,
    k_cache,
    v_cache,
    seq_lens,
    block_table,
    prefix_lens,
    window_size,
    softmax_scale=None,
    *,
    out=None,
    prefix_splits=0,
):
    """Decode one query per sequence using prefix and tail attention plus LSE merge.

    ``q``/``out`` are [B, Hq, D]; caches are strided [pages, block, Hkv, D].
    Lengths and block tables are contiguous int32 CUDA metadata. The framework
    supplies disjoint prefix counts:
    ``prefix_lens = min(original_prefix, max(seq_lens - window_size, 0))``.
    This represents R-SWA without double-counting early generated positions.
    Cache/Q inputs remain unchanged. Extra masks, quantization and prefill
    queries are outside this API. The initial vendor path supports FP16/BF16,
    D128, positive multiples-of-16 page sizes, and MetaX MACA FlashAttention.
    """
    if runtime.device.vendor_name != "metax":
        raise NotImplementedError("Paged prefix/window attention requires MetaX")
    from flaggems_vllm.runtime.backend._metax.ops.paged_prefix_window_attention import (
        paged_prefix_window_attention as implementation,
    )

    return implementation(
        q,
        k_cache,
        v_cache,
        seq_lens,
        block_table,
        prefix_lens,
        window_size,
        softmax_scale,
        out=out,
        prefix_splits=prefix_splits,
    )
