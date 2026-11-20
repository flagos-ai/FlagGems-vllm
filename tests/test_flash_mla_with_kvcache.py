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

"""
Tests for flash_mla_with_kvcache Triton implementation.
Uses the CUDA FlashMLA implementation from vLLM as ground truth.
"""

import math

import pytest
import torch

from . import conftest as cfg

# CUDA reference
try:
    from vllm.third_party.flashmla.flash_mla_interface import (
        flash_mla_with_kvcache as cuda_flash_mla,
    )
    from vllm.third_party.flashmla.flash_mla_interface import (
        get_mla_metadata as cuda_get_mla_metadata,
    )

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Triton implementation under test
import flaggems_vllm
from flaggems_vllm.ops.flash_mla_with_kvcache import FlashMLASchedMeta
from flaggems_vllm.ops.flash_mla_with_kvcache import (
    get_mla_metadata as triton_get_mla_metadata,
)

# The operator as registered, so a backend that replaces it is what is tested.
triton_flash_mla = flaggems_vllm.flash_mla_with_kvcache

DEVICE = "cuda"
CUDA_AVAILABLE = torch.cuda.is_available()
FP8_MAX = 448.0


def generate_v32_fp8_kv_cache(
    num_pages,
    page_block_size,
    h_k=1,
    d_nope=512,
    d_rope=64,
    device=DEVICE,
    scale_fmt=None,
):
    """Generate V32 FP8 KV cache: 512 FP8 + 4 FP32 scales + 64 BF16.

    scale_fmt "ue8m0" rounds every scale up to a power of two, the way
    indexer_k_quant_and_cache does, so that the format the caller promises
    matches the cache.
    """
    total_tokens = num_pages * page_block_size

    nope_data = (
        torch.randn(total_tokens, h_k, d_nope, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    nope_flat = nope_data.reshape(-1, d_nope)
    groups = nope_flat.reshape(-1, 4, 128)
    scales = groups.float().abs().amax(dim=-1) / FP8_MAX
    scales = scales.clamp(min=1e-12)
    if scale_fmt == "ue8m0":
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    quantized = (groups.float() / scales[:, :, None]).clamp(-FP8_MAX, FP8_MAX)
    fp8_data = quantized.reshape(-1, d_nope).to(torch.float8_e4m3fn)

    rope_data = (
        torch.randn(total_tokens, h_k, d_rope, dtype=torch.bfloat16, device=device)
        * 0.1
    )

    kv_cache = torch.zeros(
        num_pages, page_block_size, h_k, 656, dtype=torch.uint8, device=device
    )
    kv_cache[:, :, :, :512] = fp8_data.view(torch.uint8).reshape(
        num_pages, page_block_size, h_k, 512
    )
    kv_cache[:, :, :, 512:528] = (
        scales.reshape(num_pages, page_block_size, h_k, 4)
        .to(torch.float32)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_k, 16)
    )
    kv_cache[:, :, :, 528:656] = (
        rope_data.reshape(num_pages, page_block_size, h_k, d_rope)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_k, 128)
    )
    return kv_cache


def generate_model1_fp8_kv_cache(
    num_pages, page_block_size, h_k=1, d_nope=448, d_rope=64, device=DEVICE
):
    """Generate MODEL1 page-oriented cache: page data first, then E8M0 scales."""
    assert h_k == 1, "FlashMLA sparse decode currently supports h_k == 1"
    total_tokens = num_pages * page_block_size
    token_data_bytes = d_nope + d_rope * 2
    scale_bytes = 8

    nope_data = (
        torch.randn(total_tokens, d_nope, dtype=torch.bfloat16, device=device) * 0.1
    )
    groups = nope_data.reshape(total_tokens, 7, 64).float()
    block_max = torch.clamp(groups.abs().amax(dim=-1), min=1e-4)
    exponent = torch.ceil(torch.log2(block_max / FP8_MAX))
    scales = torch.exp2(exponent)
    fp8_data = (
        torch.clamp(groups / scales[:, :, None], -FP8_MAX, FP8_MAX)
        .reshape(total_tokens, d_nope)
        .to(torch.float8_e4m3fn)
    )

    rope_data = (
        torch.randn(total_tokens, d_rope, dtype=torch.bfloat16, device=device) * 0.1
    )
    data_bytes = torch.empty(
        total_tokens, token_data_bytes, dtype=torch.uint8, device=device
    )
    data_bytes[:, :d_nope] = fp8_data.view(torch.uint8).reshape(total_tokens, d_nope)
    data_bytes[:, d_nope:] = rope_data.view(torch.uint8).reshape(total_tokens, 128)

    encoded_scales = torch.zeros(
        total_tokens, scale_bytes, dtype=torch.uint8, device=device
    )
    encoded_scales[:, :7] = torch.clamp(exponent + 127.0, 0, 255).to(torch.uint8)

    kv_cache = torch.zeros(
        num_pages, page_block_size, h_k, 584, dtype=torch.uint8, device=device
    )
    page_flat = kv_cache.view(num_pages, -1)
    page_flat[:, : page_block_size * token_data_bytes] = data_bytes.reshape(
        num_pages, page_block_size * token_data_bytes
    )
    page_flat[:, page_block_size * token_data_bytes :] = encoded_scales.reshape(
        num_pages, page_block_size * scale_bytes
    )
    return kv_cache


def check_close(triton_out, cuda_out, name, cos_threshold=0.99):
    diff = (triton_out.float() - cuda_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        triton_out.float().flatten(), cuda_out.float().flatten(), dim=0
    ).item()
    print(
        f"  {name}: max_diff={max_diff:.6f}, "
        f"mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.8f}"
    )
    assert cos_sim > cos_threshold, f"{name} cosine similarity too low: {cos_sim}"


def _run_cuda(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    cuda_meta, _ = cuda_get_mla_metadata()
    return cuda_flash_mla(
        q, k_cache, block_table, cache_seqlens, head_dim_v, cuda_meta, **kwargs
    )


def _run_triton(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    triton_meta, _ = triton_get_mla_metadata()
    return triton_flash_mla(
        q, k_cache, block_table, cache_seqlens, head_dim_v, triton_meta, **kwargs
    )


pytestmark = [
    pytest.mark.flash_mla_with_kvcache,
    pytest.mark.skipif(
        not (VLLM_AVAILABLE and CUDA_AVAILABLE),
        reason="vLLM FlashMLA and CUDA are required",
    ),
]


# Twenty shapes per layout, taken from decode traces: GLM-5.2 runs V32 with 64
# heads, topk 2048 and 64-token pages, DeepSeek-V4 runs MODEL1 with 64 heads,
# topk 128 and an extra cache of 512 tokens in 64-token pages or 8192 in pages
# of 2. The batch sweep covers both sides of the partition counts the backends
# schedule with; the rest vary topk, page size and head count. num_pages only
# has to hold the tokens the indices point at, so it stays small.
V32_SHAPES = [
    # (batch, h_q, topk, page_block_size, num_pages)
    (1, 64, 2048, 64, 40),
    (2, 64, 2048, 64, 40),
    (4, 64, 2048, 64, 60),
    (8, 64, 2048, 64, 60),
    (16, 64, 2048, 64, 80),
    (19, 64, 2048, 64, 80),
    (20, 64, 2048, 64, 80),
    (21, 64, 2048, 64, 80),
    (32, 64, 2048, 64, 100),
    (40, 64, 2048, 64, 100),
    (64, 64, 2048, 64, 120),
    (80, 64, 2048, 64, 120),
    (128, 64, 2048, 64, 160),
    (5, 128, 2048, 64, 80),
    (2, 64, 64, 64, 50),
    (2, 128, 128, 64, 100),
    (4, 64, 512, 16, 200),
    (8, 64, 1024, 16, 120),
    (3, 64, 4096, 64, 200),
    (12, 64, 2048, 2, 1536),
]
MODEL1_SHAPES = [
    # (batch, h_q, topk, extra_topk, extra_page_block_size)
    (1, 64, 128, 0, 0),
    (2, 64, 128, 0, 0),
    (4, 64, 128, 0, 0),
    (8, 64, 128, 0, 0),
    (16, 64, 128, 0, 0),
    (20, 64, 128, 0, 0),
    (21, 64, 128, 0, 0),
    (32, 64, 128, 0, 0),
    (64, 64, 128, 0, 0),
    (128, 64, 128, 0, 0),
    (4, 64, 512, 0, 0),
    (2, 128, 128, 0, 0),
    (1, 64, 128, 512, 64),
    (8, 64, 128, 512, 64),
    (20, 64, 128, 512, 64),
    (40, 64, 128, 512, 64),
    (128, 64, 128, 512, 64),
    (1, 64, 128, 8192, 2),
    (8, 64, 128, 8192, 2),
    (64, 64, 128, 8192, 2),
]
if cfg.QUICK_MODE:
    V32_SHAPES = V32_SHAPES[::5]
    MODEL1_SHAPES = MODEL1_SHAPES[::5]


@pytest.mark.flash_mla_with_kvcache
@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"])
@pytest.mark.parametrize("batch,h_q,topk,page_block_size,num_pages", V32_SHAPES)
def test_sparse_decode_v32_fp8(batch, h_q, topk, page_block_size, num_pages, scale_fmt):
    """V32 sparse decode uses 656-byte FP8 cache and no dynamic topk.

    Both scale formats are run: the cache is quantized the way scale_fmt says
    and scale_fmt is passed along, so a backend that dequantizes a power of two
    differently is checked against the same reference.
    """
    print("\n=== test_sparse_decode_v32_fp8 ===")
    seq_q, d_qk = 1, 576
    head_dim_v = 512

    torch.manual_seed(42)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_v32_fp8_kv_cache(
        num_pages, page_block_size, scale_fmt=scale_fmt
    )
    total_tokens = num_pages * page_block_size
    indices = torch.randint(
        0, total_tokens, (batch, seq_q, topk), dtype=torch.int32, device=DEVICE
    )
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=DEVICE)

    cuda_out_buf = torch.empty(
        batch, seq_q, h_q, head_dim_v, dtype=torch.bfloat16, device=DEVICE
    )
    triton_out_buf = torch.empty_like(cuda_out_buf)
    cuda_out, cuda_lse = _run_cuda(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        out=cuda_out_buf,
    )
    triton_out, triton_lse = _run_triton(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        out=triton_out_buf,
        scale_fmt=scale_fmt,
    )

    assert cuda_out.data_ptr() == cuda_out_buf.data_ptr()
    assert triton_out.data_ptr() == triton_out_buf.data_ptr()
    check_close(triton_out, cuda_out, "out")
    check_close(triton_lse, cuda_lse, "lse")


@pytest.mark.flash_mla_with_kvcache
@pytest.mark.parametrize(
    "batch,h_q,topk,extra_topk,extra_page_block_size", MODEL1_SHAPES
)
def test_sparse_decode_model1(batch, h_q, topk, extra_topk, extra_page_block_size):
    """MODEL1: 584-byte cache, dynamic topk_length, attn_sink, out, extra cache."""
    print("\n=== test_sparse_decode_model1 ===")
    seq_q, d_qk = 1, 512
    head_dim_v = 512
    page_block_size = 64
    num_pages = math.ceil(topk / page_block_size) + 4

    torch.manual_seed(43 + batch + extra_topk)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_model1_fp8_kv_cache(num_pages, page_block_size)
    indices = torch.randint(
        0,
        num_pages * page_block_size,
        (batch, seq_q, topk),
        dtype=torch.int32,
        device=DEVICE,
    )
    topk_length = torch.randint(1, topk + 1, (batch,), dtype=torch.int32, device=DEVICE)
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=DEVICE)
    extra = {}
    if extra_topk:
        extra_num_pages = math.ceil(extra_topk / extra_page_block_size) + 4
        extra_tokens = extra_num_pages * extra_page_block_size
        extra["extra_k_cache"] = generate_model1_fp8_kv_cache(
            extra_num_pages, extra_page_block_size
        )
        extra["extra_indices_in_kvcache"] = torch.randint(
            0,
            extra_tokens,
            (batch, seq_q, extra_topk),
            dtype=torch.int32,
            device=DEVICE,
        )
        extra["extra_topk_length"] = torch.randint(
            1, extra_topk + 1, (batch,), dtype=torch.int32, device=DEVICE
        )

    cuda_out_buf = torch.empty(
        batch, seq_q, h_q, head_dim_v, dtype=torch.bfloat16, device=DEVICE
    )
    triton_out_buf = torch.empty_like(cuda_out_buf)
    cuda_out, cuda_lse = _run_cuda(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        out=cuda_out_buf,
        **extra,
    )
    triton_out, triton_lse = _run_triton(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        out=triton_out_buf,
        **extra,
    )

    assert cuda_out.data_ptr() == cuda_out_buf.data_ptr()
    assert triton_out.data_ptr() == triton_out_buf.data_ptr()
    check_close(triton_out, cuda_out, "out", cos_threshold=0.98)
    check_close(triton_lse, cuda_lse, "lse", cos_threshold=0.98)


def test_dense_decode_seq_q1():
    """Dense decode is covered only for seq_q=1 in the Triton implementation."""
    print("\n=== test_dense_decode_seq_q1 ===")
    batch, seq_q, h_q, d_qk = 4, 1, 128, 576
    h_k = 1
    head_dim_v = 512
    page_block_size = 64
    seqlen = 256

    torch.manual_seed(45)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    kv_cache = (
        torch.randn(
            total_pages, page_block_size, h_k, d_qk, dtype=torch.bfloat16, device=DEVICE
        )
        * 0.1
    )
    block_table = torch.arange(total_pages, dtype=torch.int32, device=DEVICE).view(
        batch, max_pages_per_seq
    )
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=DEVICE)
    cache_seqlens[0] = seqlen // 2
    cache_seqlens[-1] = seqlen + page_block_size

    cuda_out, cuda_lse = _run_cuda(
        q, kv_cache, block_table, cache_seqlens, head_dim_v, causal=True
    )
    triton_out, triton_lse = _run_triton(
        q, kv_cache, block_table, cache_seqlens, head_dim_v, causal=True
    )

    check_close(triton_out, cuda_out, "out")
    check_close(triton_lse, cuda_lse, "lse")


def test_error_v32_rejects_topk_length():
    q = torch.randn(1, 1, 64, 576, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_v32_fp8_kv_cache(4, 64)
    indices = torch.randint(0, 256, (1, 1, 64), dtype=torch.int32, device=DEVICE)
    topk_length = torch.ones(1, dtype=torch.int32, device=DEVICE)
    meta, _ = triton_get_mla_metadata()

    with pytest.raises(AssertionError, match="dynamic topk length"):
        triton_flash_mla(
            q,
            kv_cache,
            None,
            None,
            512,
            meta,
            is_fp8_kvcache=True,
            indices=indices,
            topk_length=topk_length,
        )


def test_error_model1_extra_cache_requires_extra_indices():
    q = torch.randn(1, 1, 64, 512, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_model1_fp8_kv_cache(4, 64)
    extra_k_cache = generate_model1_fp8_kv_cache(4, 64)
    indices = torch.randint(0, 256, (1, 1, 64), dtype=torch.int32, device=DEVICE)
    meta, _ = triton_get_mla_metadata()

    with pytest.raises(AssertionError, match="extra_indices_in_kvcache"):
        triton_flash_mla(
            q,
            kv_cache,
            None,
            None,
            512,
            meta,
            is_fp8_kvcache=True,
            indices=indices,
            extra_k_cache=extra_k_cache,
        )


def test_error_dense_rejects_sparse_only_args():
    q = torch.randn(1, 1, 64, 576, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.randn(4, 64, 1, 576, dtype=torch.bfloat16, device=DEVICE)
    block_table = torch.arange(4, dtype=torch.int32, device=DEVICE).view(1, 4)
    cache_seqlens = torch.full((1,), 64, dtype=torch.int32, device=DEVICE)
    attn_sink = torch.randn(64, dtype=torch.float32, device=DEVICE)
    meta, _ = triton_get_mla_metadata()

    with pytest.raises(AssertionError, match="must be None when dense"):
        triton_flash_mla(
            q,
            kv_cache,
            block_table,
            cache_seqlens,
            512,
            meta,
            attn_sink=attn_sink,
        )


def _sched_meta(**overrides):
    config = dict(
        b=1,
        s_q=1,
        h_q=64,
        head_dim_k=512,
        page_block_size=64,
        h_k=1,
        causal=False,
        is_fp8_kvcache=True,
        topk=64,
        extra_page_block_size=None,
        extra_topk=None,
    )
    config.update(overrides)
    return FlashMLASchedMeta(
        have_initialized=True, config=FlashMLASchedMeta.Config(**config)
    )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"b": 2}, "sched_meta.config.b"),
        # A schedule is only reusable for the layout it was built for: 512 and
        # 576 are different kernels with different tiles.
        ({"head_dim_k": 576}, "sched_meta.config.head_dim_k"),
        ({"topk": 128}, "sched_meta.config.topk"),
    ],
)
def test_error_sched_meta_reuse_mismatch(overrides, message):
    q = torch.empty(1, 1, 64, 512, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.empty(4, 64, 1, 584, dtype=torch.uint8, device=DEVICE)
    indices = torch.zeros(1, 1, 64, dtype=torch.int32, device=DEVICE)

    with pytest.raises(AssertionError, match=message):
        triton_flash_mla(
            q,
            kv_cache,
            None,
            None,
            512,
            _sched_meta(**overrides),
            is_fp8_kvcache=True,
            indices=indices,
        )


if __name__ == "__main__":
    for shape in V32_SHAPES:
        test_sparse_decode_v32_fp8(*shape)
    for shape in MODEL1_SHAPES:
        test_sparse_decode_model1(*shape)
    test_dense_decode_seq_q1()
