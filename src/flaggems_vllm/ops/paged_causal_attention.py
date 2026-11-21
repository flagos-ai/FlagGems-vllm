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

"""Variable-length causal attention over a read-only paged KV cache."""

from flaggems_vllm import runtime


def is_paged_causal_attention_available():
    if runtime.device.vendor_name != "metax":
        return False
    from flaggems_vllm.runtime.backend._metax.ops.paged_causal_attention import (
        is_paged_causal_attention_available as available,
    )

    return available()


def paged_causal_attention(
    q,
    k_cache,
    v_cache,
    query_start_loc,
    seq_lens,
    block_table,
    max_query_len,
    max_seq_len,
    softmax_scale=None,
    *,
    out=None,
):
    """Bottom-right causal attention for ragged queries and paged K/V.

    Q/out are [T,Hq,128], K/V are strided [pages,block,Hkv,128]. Device
    cumulative query offsets start at zero and terminate at T; seq_lens are
    the actual cache lengths including the current query. The framework
    owns these int32 metadata and must ensure monotone offsets and L >= Q.
    Inputs/cache are unchanged. This inference-only API supports MetaX,
    FP16/BF16, multiples-of-16 cache pages and unquantized attention without
    extra masks, biases, dropout or score caps.
    """
    if runtime.device.vendor_name != "metax":
        raise NotImplementedError("Paged causal attention requires MetaX")
    from flaggems_vllm.runtime.backend._metax.ops.paged_causal_attention import (
        paged_causal_attention as implementation,
    )

    return implementation(
        q,
        k_cache,
        v_cache,
        query_start_loc,
        seq_lens,
        block_table,
        max_query_len,
        max_seq_len,
        softmax_scale,
        out=out,
    )
