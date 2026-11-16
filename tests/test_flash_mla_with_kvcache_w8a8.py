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

import importlib
import math

import pytest
import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import HAS_TLE

from . import conftest as cfg
from .accuracy_utils import gems_assert_close, gems_assert_equal, to_reference

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

OUTPUT_ATOL = 2e-3
SUPPORTED = (
    HAS_TLE and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
)


def quantize_dense_fp8(x, *, dequantize=True):
    scale = x[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    nope = (x[..., :512].float() / scale).to(torch.float8_e4m3fn)
    rope = (x[..., 512:].float() / scale).to(torch.bfloat16)
    if dequantize:
        native = (torch.cat((nope.float(), rope.float()), -1) * scale).to(
            torch.bfloat16
        )
    else:
        native = None
    return nope, rope, scale, native


def torch_reference(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v,
    *,
    softmax_scale=None,
    causal=False,
):
    batch, queries, _, dim = q.shape
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    cache = k_cache[block_table.long()].reshape(batch, -1, dim).float()
    tokens = torch.arange(cache.shape[1], device=q.device)
    valid = tokens[None, :] < cache_seqlens[:, None]
    cache = cache.masked_fill(~valid[:, :, None], 0)
    scores = torch.einsum("bqhd,bnd->bqhn", q.float(), cache) * scale
    allowed = valid[:, None, :]
    if causal:
        query_positions = (
            cache_seqlens[:, None]
            - queries
            + torch.arange(queries, device=q.device)[None, :]
        )
        allowed = allowed & (tokens[None, None, :] <= query_positions[:, :, None])
    else:
        pass
    scores = scores.masked_fill(~allowed[:, :, None, :], -torch.inf)
    probabilities = torch.where(allowed[:, :, None, :], scores.softmax(-1), 0)
    output = torch.einsum("bqhn,bnd->bqhd", probabilities, cache[..., :head_dim_v]).to(
        q.dtype
    )
    return output, torch.logsumexp(scores, -1).transpose(1, 2)


def vllm_reference(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    metadata, _ = cuda_get_mla_metadata()
    return cuda_flash_mla(
        q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, **kwargs
    )


def assert_dense_fp8_close(actual, reference):
    reference = tuple(to_reference(tensor) for tensor in reference)
    if reference[0].device.type == "cpu":
        actual = tuple(tensor.cpu() for tensor in actual)
    else:
        pass
    gems_assert_close(actual[0], reference[0], torch.bfloat16, atol=OUTPUT_ATOL)
    gems_assert_close(actual[1], reference[1], torch.float32, atol=2e-4)


REFERENCES = (torch_reference, vllm_reference) if VLLM_AVAILABLE else (torch_reference,)


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(not SUPPORTED, reason="FP8 MLA requires NVIDIA Hopper and TLE")
@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize(
    "batch,seqlen,varlen",
    [
        (4, 128, False),
        (4, 256, True),
        (2, 33280, False),
        (8, 33280, False),
        (16, 8192, False),
    ],
)
@pytest.mark.parametrize("seq_q", (1, 2))
@pytest.mark.parametrize("h_q", (16, 32, 64, 128))
def test_dense_decode(reference, seq_q, h_q, batch, seqlen, varlen):
    head_dim_v, page_block_size = 512, 64
    torch.manual_seed(45)
    q = torch.randn(
        batch, seq_q, h_q, 576, dtype=torch.bfloat16, device=flaggems_vllm.device
    )
    max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    cache = (
        torch.randn(
            total_pages,
            page_block_size,
            1,
            576,
            dtype=torch.bfloat16,
            device=flaggems_vllm.device,
        )
        * 0.1
    )
    table = torch.arange(
        total_pages, dtype=torch.int32, device=flaggems_vllm.device
    ).view(batch, max_pages_per_seq)
    host_lengths = [seqlen] * batch
    if varlen:
        host_lengths[0] = seqlen // 2
        host_lengths[-1] = seqlen + page_block_size
    else:
        pass
    lengths = torch.tensor(host_lengths, dtype=torch.int32, device=flaggems_vllm.device)
    q_nope, q_rope, q_scale, native_q = quantize_dense_fp8(q)
    k_nope, k_rope, k_scale, native_cache = quantize_dense_fp8(cache)
    actual = flaggems_vllm.flash_mla_with_kvcache_w8a8(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        table,
        lengths,
        head_dim_v,
        causal=True,
    )
    expected = reference(
        native_q, native_cache, table, lengths, head_dim_v, causal=True
    )
    assert_dense_fp8_close(actual, expected)


def quantize_mla_content(tensor):
    scale = (
        tensor[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    )
    content = (
        (tensor[..., :512].float() / scale).round().clamp(-127, 127).to(torch.int8)
    )
    rope = (tensor[..., 512:].float() / scale).to(tensor.dtype)
    return content, rope, scale


def make_dense_int8_inputs(
    batch,
    heads,
    length,
    dtype=torch.bfloat16,
    *,
    ragged=False,
    strided=False,
    magnitude=0.3,
    extra_pages=0,
    queries=1,
    seed=42,
):
    torch.manual_seed(seed)
    pages = max(1, math.ceil(length / 64)) + extra_pages
    query = (
        torch.randn(
            batch, queries, heads, 576, dtype=dtype, device=flaggems_vllm.device
        )
        * magnitude
    )
    cache = (
        torch.randn(batch * pages, 64, 576, dtype=dtype, device=query.device)
        * magnitude
    )
    q, qr, qs = quantize_mla_content(query)
    kv, kr, ks = quantize_mla_content(cache)
    table = (
        torch.arange(batch * pages, dtype=torch.int32, device=query.device)
        .reshape(batch, pages)
        .flip(1)
    )
    lengths = torch.full((batch,), length, dtype=torch.int32, device=query.device)
    if ragged:
        lengths = torch.tensor(
            [0 if i == 0 else max(1, length - i * 17) for i in range(batch)],
            device=query.device,
            dtype=torch.int32,
        )
    if strided:
        outer_views = []
        for tensor in (q, qr, kv, kr, qs, ks, table):
            storage = torch.empty(
                (tensor.shape[0] * 2, *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            view = storage[::2]
            view.copy_(tensor)
            outer_views.append(view)
        q, qr, kv, kr, qs, ks, table = outer_views
    return (q, qr, kv, kr, qs, ks, table, lengths)


def dense_int8_mla_reference(tensors, causal=False):
    q, qr, kv, kr, qs, ks, table, lengths = tensors
    query = torch.cat((q.float(), qr.float()), -1) * qs
    cache = torch.cat((kv.float(), kr.float()), -1) * ks
    return dense_mla_reference(query, cache, table, lengths, causal)


def dense_mla_reference(query, cache, table, lengths, causal=False):
    batch, queries, heads, _ = query.shape
    output = torch.zeros(
        (batch, queries, heads, 512), dtype=torch.float32, device=query.device
    )
    lse = torch.full(
        (batch, heads, queries), float("inf"), dtype=torch.float32, device=query.device
    )
    # Bound the FP32 reference workspace to one request for long-cache benchmarks.
    for row in range(batch):
        length = int(lengths[row].item())
        if length == 0:
            continue
        row_output, row_lse = torch_reference(
            query[row : row + 1].float(),
            cache,
            table[row : row + 1, : math.ceil(length / 64)],
            lengths[row : row + 1],
            512,
            causal=causal,
        )
        output[row] = row_output[0]
        lse[row] = torch.where(torch.isneginf(row_lse[0]), float("inf"), row_lse[0])
    return output, lse


INT8_OP_NAME = "flash_mla_with_kvcache_fwd_w8a8_int8"
INT8_VENDOR_MODULE = "flaggems_vllm.runtime.backend._thead.ops." + INT8_OP_NAME
INT8_CUSTOMIZED_OPS = dict(flaggems_vllm.runtime.backend.get_customized_ops())
INT8_SUPPORTED = INT8_OP_NAME in INT8_CUSTOMIZED_OPS
PPU_INT8_SUPPORTED = (
    INT8_SUPPORTED
    and INT8_CUSTOMIZED_OPS[INT8_OP_NAME].__module__ == INT8_VENDOR_MODULE
)
INT8_CASES = (
    [(1, 64, 65), (3, 17, 129)]
    if cfg.QUICK_MODE
    else [
        (2, 64, 0),
        (1, 1, 1),
        (1, 64, 63),
        (1, 64, 64),
        (1, 64, 65),
        (3, 17, 129),
        (2, 128, 640),
        (1, 64, 8192),
    ]
)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(not INT8_SUPPORTED, reason="backend has no matching INT8 MLA path")
@pytest.mark.parametrize("batch,heads,length", INT8_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [1, 4])
def test_flash_mla_with_kvcache_fwd_w8a8_int8(batch, heads, length, dtype, splits):
    tensors = make_dense_int8_inputs(
        batch, heads, length, dtype, ragged=batch > 2, strided=True
    )
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, num_splits=splits
    )
    expected, expected_lse = dense_int8_mla_reference(tensors)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(not INT8_SUPPORTED, reason="backend has no matching INT8 MLA path")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_mla_int8_large_head_tiles(dtype):
    tensors = make_dense_int8_inputs(128, 128, 257, dtype)
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, num_splits=1
    )
    expected, expected_lse = dense_int8_mla_reference(tensors)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(not INT8_SUPPORTED, reason="backend has no matching INT8 MLA path")
def test_flash_mla_int8_concentrated_probability():
    tensors = make_dense_int8_inputs(2, 64, 257, magnitude=2.0)
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, num_splits=4
    )
    expected, expected_lse = dense_int8_mla_reference(tensors)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.02)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(not INT8_SUPPORTED, reason="backend has no matching INT8 MLA path")
@pytest.mark.parametrize("batch,heads", [(0, 64), (2, 0)])
@pytest.mark.parametrize("queries", [1, 2])
def test_flash_mla_int8_empty_batch(batch, heads, queries):
    tensors = make_dense_int8_inputs(batch, heads, 64, queries=queries)
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(*tensors)
    assert output.shape == (batch, queries, heads, 512) and lse.shape == (
        batch,
        heads,
        queries,
    )


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(not INT8_SUPPORTED, reason="backend has no matching INT8 MLA path")
@pytest.mark.parametrize("queries", [1, 2])
def test_flash_mla_int8_prepared_length_update(queries):
    tensors = make_dense_int8_inputs(2, 64, 257, queries=queries)
    handle, _ = flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, causal=True
    )
    tensors[-1].fill_(65)
    output, lse = handle()
    expected, expected_lse = dense_int8_mla_reference(tensors, True)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(not INT8_SUPPORTED, reason="backend has no matching INT8 MLA path")
def test_flash_mla_int8_large_cache_address():
    tensors = make_dense_int8_inputs(1, 64, 64)
    q, qr, kv, kr, qs, ks, table, lengths = tensors
    pages = 65538
    cache = torch.empty((pages, 64, 512), dtype=kv.dtype, device=kv.device)
    rope_cache = torch.empty((pages, 64, 64), dtype=kr.dtype, device=kr.device)
    scales = torch.empty((pages, 64, 1), dtype=ks.dtype, device=ks.device)
    cache[-1].copy_(kv[0])
    rope_cache[-1].copy_(kr[0])
    scales[-1].copy_(ks[0])
    high_page = torch.full_like(table, pages - 1)
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        q, qr, cache, rope_cache, qs, scales, high_page, lengths
    )
    expected, expected_lse = dense_int8_mla_reference(tensors)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(not INT8_SUPPORTED, reason="backend has no matching INT8 MLA path")
def test_flash_mla_int8_prefill_is_unsupported():
    tensors = list(make_dense_int8_inputs(1, 64, 64))
    tensors[0] = tensors[0].expand(1, 3, 64, 512)
    with pytest.raises(NotImplementedError, match="requires Q"):
        flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(*tensors)


@triton.jit
def probability_quantization_kernel(
    Input,
    Coarse,
    Residual,
    N: tl.constexpr,
    QUANTIZER: tl.constexpr,
    USE_PTX: tl.constexpr,
    RADIX: tl.constexpr = 127,
):
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    probability = tl.load(Input + offsets, offsets < N, 0.0)
    if USE_PTX:
        if RADIX == 256:
            coarse, residual = QUANTIZER(probability, NORMALIZED=True, RADIX=RADIX)
        else:
            coarse, residual = QUANTIZER(probability)
    else:
        if RADIX == 256:
            quantized = tl.floor(probability * 256.0 + 0.5).to(tl.int32)
            coarse = ((quantized + 128) >> 8).to(tl.int8)
            residual = quantized.to(tl.int8)
        else:
            coarse = tl.floor(probability + 0.5).to(tl.int8)
            residual = tl.floor((probability - coarse.to(tl.float32)) * 127.0 + 0.5).to(
                tl.int8
            )
    tl.store(Coarse + offsets, coarse, offsets < N)
    tl.store(Residual + offsets, residual, offsets < N)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(
    not PPU_INT8_SUPPORTED, reason="backend has no matching INT8 MLA path"
)
@pytest.mark.parametrize("radix", [127, 256])
def test_flash_mla_int8_probability_quantization(radix):
    from flaggems_vllm.runtime.backend._thead.ops.flash_mla_with_kvcache_fwd_w8a8_int8 import (
        quantize_mla_probabilities,
        quantize_mla_probabilities_packed,
    )

    torch.manual_seed(42)
    device = flaggems_vllm.device
    if radix == 256:
        halves = (torch.arange(32513, device=device, dtype=torch.float32) + 0.5) / 256
    else:
        halves = torch.arange(127, device=device, dtype=torch.float32) + 0.5
    boundaries = torch.cat(
        (
            torch.nextafter(halves, torch.full_like(halves, -float("inf"))),
            halves,
            torch.nextafter(halves, torch.full_like(halves, float("inf"))),
        )
    )
    residual_halves = (
        64 + (torch.arange(-63, 64, device=device, dtype=torch.float32) + 0.5) / 127
    )
    small = torch.tensor(
        [0, 1, 0x007FFFFF, 0x00800000], dtype=torch.int32, device=device
    ).view(torch.float32)
    probabilities = torch.cat(
        (
            torch.rand(262144, device=device) * 127,
            boundaries,
            residual_halves,
            small,
            torch.tensor([127.0], device=device),
        )
    )
    coarse_reference = torch.empty_like(probabilities, dtype=torch.int8)
    residual_reference = torch.empty_like(coarse_reference)
    coarse = torch.empty_like(coarse_reference)
    residual = torch.empty_like(coarse_reference)
    grid = (triton.cdiv(probabilities.numel(), 1024),)
    probability_quantization_kernel[grid](
        probabilities,
        coarse_reference,
        residual_reference,
        probabilities.numel(),
        quantize_mla_probabilities,
        False,
        radix,
    )
    quantizers = (
        (quantize_mla_probabilities, quantize_mla_probabilities_packed)
        if radix == 127
        else (quantize_mla_probabilities_packed,)
    )
    for quantizer in quantizers:
        probability_quantization_kernel[grid](
            probabilities,
            coarse,
            residual,
            probabilities.numel(),
            quantizer,
            True,
            radix,
        )
        gems_assert_equal(coarse, coarse_reference)
        gems_assert_equal(residual, residual_reference)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(not INT8_SUPPORTED, reason="backend has no matching INT8 MLA path")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [1, 4])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_mla_int8_two_queries(dtype, splits, causal):
    tensors = make_dense_int8_inputs(
        3, 17, 129, dtype, queries=2, ragged=True, strided=True
    )
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, causal=causal, num_splits=splits
    )
    expected, expected_lse = dense_int8_mla_reference(tensors, causal)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(not INT8_SUPPORTED, reason="backend has no matching INT8 MLA path")
@pytest.mark.parametrize("length", [0, 1, 2, 65])
def test_flash_mla_int8_causal_query_boundaries(length):
    tensors = make_dense_int8_inputs(1, 16, length, queries=2)
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, causal=True, num_splits=4
    )
    expected, expected_lse = dense_int8_mla_reference(tensors, True)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(not INT8_SUPPORTED, reason="backend has no matching INT8 MLA path")
@pytest.mark.parametrize("queries", [1, 2])
def test_flash_mla_int8_nan_cache_tail(queries):
    tensors = make_dense_int8_inputs(2, 32, 65, queries=queries)
    # The page table reverses physical pages, so logical positions 65..127 are in page zero.
    for row in range(2):
        page = row * 2
        tensors[3][page, 1:].fill_(float("nan"))
        tensors[5][page, 1:].fill_(float("nan"))
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, causal=True, num_splits=4
    )
    expected, expected_lse = dense_int8_mla_reference(tensors, True)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(
    not PPU_INT8_SUPPORTED, reason="backend has no matching INT8 MLA path"
)
@pytest.mark.parametrize(
    "batch,heads,queries,splits,varlen",
    [(128, 16, 1, 1, True), (65, 17, 2, 4, True), (128, 16, 1, 1, False)],
)
def test_flash_mla_int8_request_order(
    monkeypatch, batch, heads, queries, splits, varlen
):
    vendor = importlib.import_module(INT8_VENDOR_MODULE)
    kernel = vendor.dense_int8_mla_kernel.jit_function

    class SortedKernel:
        def __getitem__(self, grid):
            def launch(*args):
                return kernel[grid({"BH": 16})](
                    *args,
                    BH=16,
                    BK=128,
                    FUSED_PV=True,
                    PAIR_PV=True,
                    QK_SPLITS=4,
                    COLUMN_SOFTMAX=False,
                    SORT_REQUESTS=True,
                    num_warps=8,
                    num_stages=1,
                )

            return launch

    monkeypatch.setattr(vendor, "dense_int8_mla_kernel", SortedKernel())
    tensors = make_dense_int8_inputs(
        batch, heads, 257, queries=queries, strided=True, seed=7531
    )
    if varlen:
        lengths = [(row * 73) % 258 for row in range(batch)]
    else:
        lengths = [257] * batch
    tensors[7].copy_(
        torch.tensor(
            lengths,
            dtype=torch.int32,
            device=tensors[7].device,
        )
    )
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, num_splits=splits, causal=True
    )
    expected, expected_lse = dense_int8_mla_reference(tensors, causal=True)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(
    not PPU_INT8_SUPPORTED, reason="backend has no matching INT8 MLA path"
)
@pytest.mark.parametrize("batch,varlen", [(65, False), (128, True)])
def test_flash_mla_int8_auto_splits(batch, varlen):
    tensors = make_dense_int8_inputs(
        batch, 16, 4097, ragged=varlen, strided=True, seed=9327
    )
    expected, expected_lse = dense_int8_mla_reference(tensors, causal=True)
    for splits in [None, 1, 4]:
        handle, (output, lse) = (
            flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
                *tensors, causal=True, num_splits=splits
            )
        )
        gems_assert_close(output, expected, torch.bfloat16, atol=0.002)
        gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)
        assert splits is None or handle.splits == splits
        replay, replay_lse = handle()
        gems_assert_close(replay, expected, torch.bfloat16, atol=0.002)
        gems_assert_close(replay_lse, expected_lse, torch.float32, atol=2e-4)
