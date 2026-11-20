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

import math

import pytest
import torch

import flaggems_vllm
from tests.flash_mla_with_kvcache_w8a8_utils import (
    DenseW8A8Inputs,
    assert_dense_fp8_close,
    get_backend_adapter,
)

from .accuracy_utils import to_reference

ADAPTER = get_backend_adapter()
SUPPORTED = ADAPTER is not None and ADAPTER.supported()


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(
    not SUPPORTED, reason="No registered quantized dense MLA implementation"
)
@pytest.mark.parametrize("seq_q", (1, 2))
@pytest.mark.parametrize("h_q", (16, 32, 64, 128))
def test_dense_decode(seq_q, h_q):
    batch, d_qk = 4, 576
    h_k = 1
    head_dim_v = 512
    page_block_size = 64
    seqlen = 256

    torch.manual_seed(45)
    q = torch.randn(
        batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=flaggems_vllm.device
    )
    max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    kv_cache = (
        torch.randn(
            total_pages,
            page_block_size,
            h_k,
            d_qk,
            dtype=torch.bfloat16,
            device=flaggems_vllm.device,
        )
        * 0.1
    )
    block_table = torch.arange(
        total_pages, dtype=torch.int32, device=flaggems_vllm.device
    ).view(batch, max_pages_per_seq)
    cache_seqlens = torch.full(
        (batch,), seqlen, dtype=torch.int32, device=flaggems_vllm.device
    )
    cache_seqlens[0] = seqlen // 2
    cache_seqlens[-1] = seqlen + page_block_size
    q_nope, q_rope, q_scale, native_q = ADAPTER.quantize(q, benchmark=False)
    k_nope, k_rope, k_scale, native_cache = ADAPTER.quantize(kv_cache, benchmark=False)
    host_lengths = (seqlen // 2, seqlen, seqlen, seqlen + page_block_size)
    inputs = DenseW8A8Inputs(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
        None,
        None,
        host_lengths,
        native_q,
        native_cache,
    )
    reference = tuple(to_reference(t) for t in ADAPTER.functional_reference(inputs))
    actual = flaggems_vllm.flash_mla_with_kvcache_w8a8(*inputs[:9], causal=True)
    assert_dense_fp8_close(actual, reference, output_atol=ADAPTER.OUTPUT_ATOL)
    ADAPTER.validate_functional(inputs, actual, reference)
