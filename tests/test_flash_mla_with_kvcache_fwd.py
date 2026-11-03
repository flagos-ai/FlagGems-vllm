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
from benchmark.test_flash_mla_with_kvcache_fwd import (
    assert_dense_int8_close,
    assert_dense_int8_replay,
)

from .accuracy_utils import gems_assert_close, to_reference
from .test_flash_mla import ref_mla

HAS_INT8_MLA = "flash_mla_with_kvcache_fwd_w8a8_int8" in dict(
    flaggems_vllm.runtime.backend.get_customized_ops()
)


@pytest.mark.flash_mla_with_kvcache_fwd
@pytest.mark.skipif(not HAS_INT8_MLA, reason="Requires registered INT8 MLA")
@pytest.mark.parametrize(
    "heads,queries", ((32, 1), (16, 2), (64, 1), (32, 2), (128, 1), (64, 2), (128, 2))
)
@pytest.mark.parametrize(
    "rope_dtype,scale", ((torch.bfloat16, 10000000000.0), (torch.float16, 10000.0))
)
def test_dense_int8_rope_cancellation(heads, queries, rope_dtype, scale):
    batch = 128
    query = torch.zeros((batch, queries, heads, 512), dtype=torch.int8, device="cuda")
    query[..., 0] = 1
    query_rope = torch.zeros(
        (batch, queries, heads, 64), dtype=rope_dtype, device="cuda"
    )
    query_rope[..., 0] = 16384
    query_rope[..., 16] = 16384
    cache = torch.zeros((1, 64, 1, 512), dtype=torch.int8, device="cuda")
    cache[0, 0, 0, 0] = 1
    cache[0, 1, 0, 1] = 127
    cache_rope = torch.zeros((1, 64, 1, 64), dtype=rope_dtype, device="cuda")
    cache_rope[0, :2, 0, 0] = 16384
    cache_rope[0, :2, 0, 16] = -16384
    query_scale = torch.ones(
        (batch, queries, heads, 1), dtype=torch.float32, device="cuda"
    )
    cache_scale = torch.full((1, 64, 1, 1), scale, dtype=torch.float32, device="cuda")
    table = torch.zeros((batch, 64), dtype=torch.int32, device="cuda")
    lengths = torch.full((batch,), 2, dtype=torch.int32, device="cuda")
    lengths[0] = 1
    expected_out = torch.zeros_like(query, dtype=rope_dtype)
    expected_out[..., 0] = scale
    expected_lse = torch.full(
        (batch, heads, queries),
        scale * 576 ** (-0.5),
        dtype=torch.float32,
        device="cuda",
    )
    if queries == 2:
        expected_out[0, 0] = 0
        expected_lse[0, :, 0] = float("inf")
    else:
        pass
    actual = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        query,
        query_rope,
        cache,
        cache_rope,
        query_scale,
        cache_scale,
        table,
        lengths,
        512,
        causal=True,
    )
    gems_assert_close(actual[0], to_reference(expected_out), rope_dtype, atol=0.002)
    gems_assert_close(actual[1], to_reference(expected_lse), torch.float32, atol=0.0002)


@pytest.mark.flash_mla_with_kvcache_fwd
@pytest.mark.skipif(
    not HAS_INT8_MLA or flaggems_vllm.runtime.device.vendor_name != "hygon",
    reason="Requires Hygon buffer addressing",
)
@pytest.mark.parametrize("large_kv", (True, False))
def test_dense_int8_large_buffer_offsets_hygon(large_kv):
    batch, queries, heads, capacity = (
        (64, 1, 64, 65) if large_kv else (128, 2, 128, 520)
    )
    first_page = 2**31 // (64 * 512) if large_kv else 0
    pages = first_page + 2
    cache = torch.empty((pages, 64, 1, 512), dtype=torch.int8, device="cuda")
    cache_rope = torch.empty((pages, 64, 1, 64), dtype=torch.bfloat16, device="cuda")
    cache_scale = torch.empty((pages, 64, 1, 1), dtype=torch.float32, device="cuda")
    features = (torch.arange(512, device="cuda") % 17 - 8).to(torch.int8)
    cache[first_page:].copy_(features)
    cache_rope[first_page:].zero_()
    cache_scale[first_page:].fill_(1.0)
    query = torch.zeros((batch, queries, heads, 512), dtype=torch.int8, device="cuda")
    query_rope = torch.zeros(
        (batch, queries, heads, 64), dtype=torch.bfloat16, device="cuda"
    )
    query_scale = torch.ones(
        (batch, queries, heads, 1), dtype=torch.float32, device="cuda"
    )
    table = (
        torch.arange(batch * capacity, device="cuda").reshape(batch, capacity) % 2
        + first_page
    ).to(torch.int32)
    lengths = torch.tensor(
        [0, 1, 63, 64, 65, 126, 127, 128], dtype=torch.int32, device="cuda"
    ).repeat(batch // 8)
    actual = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        query,
        query_rope,
        cache,
        cache_rope,
        query_scale,
        cache_scale,
        table,
        lengths,
        512,
        causal=True,
        num_splits=1,
    )
    attended = (
        lengths[:, None] - queries + torch.arange(queries, device="cuda") + 1
    ).clamp_min(0)
    expected_out = (
        torch.where(attended[:, :, None, None] > 0, features[None, None, None, :], 0)
        .expand(batch, queries, heads, 512)
        .to(torch.bfloat16)
    )
    expected_lse = torch.where(
        attended > 0, attended.float().clamp_min(1).log(), float("inf")
    )[:, None, :].expand(batch, heads, queries)
    gems_assert_close(actual[0], to_reference(expected_out), torch.bfloat16, atol=0.002)
    gems_assert_close(actual[1], to_reference(expected_lse), torch.float32, atol=0.0002)


def run_reference(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    return ref_mla(
        q,
        block_table,
        k_cache,
        block_table.shape[1] * 64,
        64,
        q.shape[0],
        q.shape[1],
        cache_seqlens,
        q.shape[2],
        1,
        q.shape[-1],
        head_dim_v,
        kwargs["causal"],
    )


def quantize_dense_int8(x, d_nope=512):
    """Per-token quantized content, scaled BF16 RoPE and BF16 dequantization."""
    peak = 127.0
    scale = x[..., :d_nope].float().abs().amax(-1, keepdim=True) / peak
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    nope = (x[..., :d_nope].float() / scale).round().to(torch.int8)
    rope = (x[..., d_nope:].float() / scale).to(torch.bfloat16)
    dequantized = torch.cat((nope.float(), rope.float()), -1) * scale
    return (nope, rope, scale, dequantized.to(torch.bfloat16))


@pytest.mark.flash_mla_with_kvcache_fwd
@pytest.mark.skipif(not HAS_INT8_MLA, reason="Requires registered dense MLA")
@pytest.mark.parametrize("seq_q", (1, 2))
@pytest.mark.parametrize("h_q", (16, 32, 64, 128))
def test_dense_decode(seq_q, h_q):
    batch, d_qk = (4, 576)
    h_k = 1
    head_dim_v = 512
    page_block_size = 64
    seqlen = 256
    torch.manual_seed(45)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device="cuda")
    max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    kv_cache = (
        torch.randn(
            total_pages, page_block_size, h_k, d_qk, dtype=torch.bfloat16, device="cuda"
        )
        * 0.1
    )
    block_table = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(
        batch, max_pages_per_seq
    )
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device="cuda")
    cache_seqlens[0] = seqlen // 2
    cache_seqlens[-1] = seqlen + page_block_size
    q_nope, q_rope, q_scale, q = quantize_dense_int8(q)
    k_nope, k_rope, k_scale, kv_cache = quantize_dense_int8(kv_cache)
    cuda_out, cuda_lse = run_reference(
        q, kv_cache, block_table, cache_seqlens, head_dim_v, causal=True
    )
    cuda_out, cuda_lse = (to_reference(cuda_out), to_reference(cuda_lse))
    inputs = (
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
    )
    out, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(*inputs, causal=True)
    assert_dense_int8_close((out, lse), (cuda_out, cuda_lse))
    handle, _ = flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
        *inputs, causal=True
    )
    assert_dense_int8_replay(handle, (cuda_out, cuda_lse))
    return
