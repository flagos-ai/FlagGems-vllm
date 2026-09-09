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

"""Repository-native benchmarks for the Qwen4 self-developed kernels.

The three model-vendor QSA baselines retain their original ownership. Their
source is ``flagos-ai/vllm-plugin-FL``'s Qwen3.8-Flash-Next
``gpu/ops/qsa.py`` snapshot; exact provenance is recorded by
``QWEN4_VENDOR_QSA_SOURCE`` in the production module.
"""

import math

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.qwen4.hyperconnection import (
    qwen4_grouped_gemma_rmsnorm,
    qwen4_hc_gate_reduce,
    qwen4_hc_inject_combine,
)
from flaggems_vllm.ops.qwen4.ple_state import ple_state_gather, ple_state_scatter_
from flaggems_vllm.ops.qwen4.qsa import (
    qwen4_compress_norm_mrope_store_groups,
    qwen4_qsa_mqa_paged_dot,
    qwen4_store_qsa_kv_rows,
    qwen4_vendor_compress_qsa_groups,
    qwen4_vendor_qsa_mqa_paged,
    qwen4_vendor_store_qsa_rows,
)

from . import base

ROWS = [(1,), (8,), (64,)]
HC = 4
HIDDEN = 2560
EPS = 1.0e-6
QSA_DIM = 128
QSA_HEADS = 4
QSA_RATIO = 4
PAGE_SIZE = 16
SEQUENCE_LENGTH = 4096
PAGES_PER_REQUEST = SEQUENCE_LENGTH // PAGE_SIZE
COMPRESSED_LENGTH = SEQUENCE_LENGTH // QSA_RATIO
COMPRESSED_PAGES_PER_REQUEST = COMPRESSED_LENGTH // PAGE_SIZE
ROTARY_DIM = 64
MROPE_SECTION = (11, 11, 10)


class Qwen4Benchmark(base.GenericBenchmark):
    """Use the common benchmark runner with Qwen4 decode row counts."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = ROWS
        self.shape_desc = "rows"


def _benchmark(op_name, torch_op, gems_op, input_fn):
    bench = Qwen4Benchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        gems_op=gems_op,
        dtypes=[torch.bfloat16],
    )
    bench.run()


def _grouped_rmsnorm_input(shape, dtype, device):
    (rows,) = shape
    torch.manual_seed(1000 + rows)
    x = torch.randn((rows, HC * HIDDEN), device=device, dtype=dtype)
    weight = torch.randn((HC * HIDDEN,), device=device, dtype=dtype) * 0.01
    yield x, weight, HC, EPS


def _torch_grouped_rmsnorm(x, weight, hc_count, eps):
    hidden = weight.numel() // hc_count
    x3 = x.reshape(-1, hc_count, hidden).float()
    weight2 = weight.reshape(hc_count, hidden).float()
    return (
        (x3 * torch.rsqrt(x3.square().mean(-1, keepdim=True) + eps) * (1.0 + weight2))
        .to(x.dtype)
        .reshape_as(x)
    )


@pytest.mark.qwen4_grouped_gemma_rmsnorm
def test_qwen4_grouped_gemma_rmsnorm():
    _benchmark(
        "qwen4_grouped_gemma_rmsnorm",
        _torch_grouped_rmsnorm,
        qwen4_grouped_gemma_rmsnorm,
        _grouped_rmsnorm_input,
    )


def _hc_gate_input(shape, dtype, device):
    (rows,) = shape
    torch.manual_seed(2000 + rows)
    logits = torch.randn((rows, HC * HIDDEN), device=device, dtype=dtype)
    normed = torch.randn_like(logits)
    yield logits, normed, HC


def _torch_hc_gate_reduce(logits, normed, hc_count):
    hidden = logits.shape[-1] // hc_count
    return (
        (
            torch.sigmoid(logits.float().reshape(-1, hc_count, hidden))
            * normed.float().reshape(-1, hc_count, hidden)
        )
        .mean(-2)
        .to(normed.dtype)
    )


@pytest.mark.qwen4_hc_gate_reduce
def test_qwen4_hc_gate_reduce():
    _benchmark(
        "qwen4_hc_gate_reduce",
        _torch_hc_gate_reduce,
        qwen4_hc_gate_reduce,
        _hc_gate_input,
    )


def _hc_inject_input(shape, dtype, device):
    (rows,) = shape
    torch.manual_seed(3000 + rows)
    injection = torch.randn((rows, HC), device=device, dtype=dtype)
    block = torch.randn((rows, HIDDEN), device=device, dtype=dtype)
    residual = torch.randn((rows, HC * HIDDEN), device=device, dtype=dtype)
    yield injection, block, residual, HC


def _torch_hc_inject(injection, block, residual, hc_count):
    alpha = 2.0 * torch.sigmoid(injection.float() / hc_count)
    return (
        (
            residual.float().reshape(-1, hc_count, block.shape[-1])
            + block.float().unsqueeze(-2) * alpha.unsqueeze(-1)
        )
        .to(residual.dtype)
        .reshape_as(residual)
    )


@pytest.mark.qwen4_hc_inject_combine
def test_qwen4_hc_inject_combine():
    _benchmark(
        "qwen4_hc_inject_combine",
        _torch_hc_inject,
        qwen4_hc_inject_combine,
        _hc_inject_input,
    )


def _qsa_mqa_input(shape, dtype, device):
    (rows,) = shape
    torch.manual_seed(4000 + rows)
    pages_per_request = COMPRESSED_PAGES_PER_REQUEST
    cache = torch.randn(
        (rows * pages_per_request, PAGE_SIZE, 1, QSA_DIM),
        device=device,
        dtype=dtype,
    )
    q = torch.randn((rows, QSA_HEADS, QSA_DIM), device=device, dtype=dtype)
    page_table = torch.arange(
        rows * pages_per_request, device=device, dtype=torch.int32
    ).reshape(rows, pages_per_request)
    requests = torch.arange(rows, device=device, dtype=torch.int32)
    positions = torch.full(
        (rows,), SEQUENCE_LENGTH - 1, device=device, dtype=torch.int64
    )
    lengths = torch.full((rows,), SEQUENCE_LENGTH, device=device, dtype=torch.int64)
    yield q, cache, page_table, requests, positions, lengths, QSA_RATIO


def _torch_qsa_mqa(q, cache, table, requests, positions, lengths, ratio):
    columns = table.shape[1] * cache.shape[1]
    request = requests.to(torch.int64)
    safe_request = request.clamp(0, table.shape[0] - 1)
    visible = torch.minimum(
        (positions + 1) // ratio,
        lengths.index_select(0, safe_request) // ratio,
    )
    column = torch.arange(columns, device=q.device, dtype=torch.int64)
    logical_page = column // cache.shape[1]
    page_offset = column % cache.shape[1]
    physical_page = table[safe_request[:, None], logical_page[None, :]].to(torch.int64)
    valid = (
        (column[None, :] < visible[:, None])
        & (physical_page >= 0)
        & (physical_page < cache.shape[0])
    )
    keys = cache[
        physical_page.clamp(0, cache.shape[0] - 1), page_offset[None, :], 0
    ].float()
    logits = torch.relu(torch.matmul(q.float(), keys.transpose(1, 2))).sum(
        1
    ) / math.sqrt(QSA_DIM)
    return logits.masked_fill(~valid, -float("inf")), visible.to(torch.int32)


@pytest.mark.qwen4_vendor_qsa_mqa_paged
def test_qwen4_vendor_qsa_mqa_paged():
    _benchmark(
        "qwen4_vendor_qsa_mqa_paged",
        _torch_qsa_mqa,
        qwen4_vendor_qsa_mqa_paged,
        _qsa_mqa_input,
    )


@pytest.mark.qwen4_qsa_mqa_paged_dot
def test_qwen4_qsa_mqa_paged_dot_vs_vendor():
    _benchmark(
        "qwen4_qsa_mqa_paged_dot_vs_vendor",
        qwen4_vendor_qsa_mqa_paged,
        qwen4_qsa_mqa_paged_dot,
        _qsa_mqa_input,
    )


def _qsa_store_input(shape, dtype, device):
    (rows,) = shape
    torch.manual_seed(5000 + rows)
    blocks = (rows + PAGE_SIZE - 1) // PAGE_SIZE + 2
    cache = torch.randn((blocks, PAGE_SIZE, 1, QSA_DIM), device=device, dtype=dtype)
    values = torch.randn((rows, 1, QSA_DIM), device=device, dtype=dtype)
    slots = torch.arange(rows, device=device, dtype=torch.int64)
    yield cache, slots, values


def _torch_qsa_store(cache, slots, values):
    cache.index_put_(
        (slots // cache.shape[1], slots % cache.shape[1]),
        values,
        accumulate=False,
    )


@pytest.mark.qwen4_vendor_store_qsa_rows
def test_qwen4_vendor_store_qsa_rows():
    _benchmark(
        "qwen4_vendor_store_qsa_rows",
        _torch_qsa_store,
        qwen4_vendor_store_qsa_rows,
        _qsa_store_input,
    )


def _qsa_kv_store_input(shape, dtype, device):
    (rows,) = shape
    torch.manual_seed(6000 + rows)
    blocks = (rows + PAGE_SIZE - 1) // PAGE_SIZE + 2
    k_cache = torch.randn((blocks, PAGE_SIZE, 1, QSA_DIM), device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    key = torch.randn((rows, 1, QSA_DIM), device=device, dtype=dtype)
    value = torch.randn_like(key)
    slots = torch.arange(rows, device=device, dtype=torch.int64)
    yield k_cache, v_cache, slots, key, value


def _vendor_qsa_kv_store(k_cache, v_cache, slots, key, value):
    qwen4_vendor_store_qsa_rows(k_cache, slots, key)
    qwen4_vendor_store_qsa_rows(v_cache, slots, value)


@pytest.mark.qwen4_store_qsa_kv_rows
def test_qwen4_store_qsa_kv_rows_vs_vendor():
    _benchmark(
        "qwen4_store_qsa_kv_rows_vs_vendor",
        _vendor_qsa_kv_store,
        qwen4_store_qsa_kv_rows,
        _qsa_kv_store_input,
    )


def _make_qsa_compress_inputs(rows, dtype, device):
    raw_blocks = rows * PAGES_PER_REQUEST
    raw_cache = torch.randn(
        (raw_blocks, PAGE_SIZE, 1, QSA_DIM), device=device, dtype=dtype
    )
    block_table = torch.arange(raw_blocks, device=device, dtype=torch.int32).reshape(
        rows, PAGES_PER_REQUEST
    )
    requests = torch.arange(rows, device=device, dtype=torch.int32)
    positions = torch.full(
        (rows,), SEQUENCE_LENGTH - 1, device=device, dtype=torch.int64
    )
    compressed_slots = (
        torch.arange(rows, device=device, dtype=torch.int64)
        * COMPRESSED_PAGES_PER_REQUEST
        * PAGE_SIZE
    )
    local_positions = torch.arange(
        SEQUENCE_LENGTH, device=device, dtype=torch.int64
    ).reshape(PAGES_PER_REQUEST, PAGE_SIZE)
    rope_cache = torch.empty(
        (raw_blocks, PAGE_SIZE, 1, 3), device=device, dtype=torch.int64
    )
    repeated_positions = local_positions.repeat(rows, 1)
    rope_cache[:, :, 0, 0] = repeated_positions
    rope_cache[:, :, 0, 1] = repeated_positions // 2
    rope_cache[:, :, 0, 2] = repeated_positions // 3
    return (
        raw_cache,
        block_table,
        requests,
        positions,
        compressed_slots,
        rope_cache,
    )


def _torch_qsa_compress(
    raw_cache,
    block_table,
    requests,
    positions,
    compressed_slots,
    ratio,
    rope_cache,
):
    _ = compressed_slots
    request = requests.to(torch.int64)
    offsets = torch.arange(ratio - 1, -1, -1, device=raw_cache.device)
    source_positions = positions[:, None] - offsets[None, :]
    logical_pages = source_positions // raw_cache.shape[1]
    page_offsets = source_positions % raw_cache.shape[1]
    physical_pages = block_table[request[:, None], logical_pages]
    pooled = (
        (
            raw_cache[physical_pages, page_offsets, 0]
            .float()
            .sum(1, dtype=torch.float32)
            / ratio
        )
        .to(raw_cache.dtype)
        .unsqueeze(1)
    )
    first_positions = positions - ratio + 1
    first_pages = first_positions // raw_cache.shape[1]
    first_offsets = first_positions % raw_cache.shape[1]
    first_physical_pages = block_table[request, first_pages]
    axes = rope_cache[first_physical_pages, first_offsets, 0, :3]
    return pooled, axes


def _qsa_compress_input(shape, dtype, device):
    (rows,) = shape
    torch.manual_seed(7000 + rows)
    raw, table, requests, positions, slots, rope = _make_qsa_compress_inputs(
        rows, dtype, device
    )
    yield raw, table, requests, positions, slots, QSA_RATIO, rope


@pytest.mark.qwen4_vendor_compress_qsa_groups
def test_qwen4_vendor_compress_qsa_groups():
    _benchmark(
        "qwen4_vendor_compress_qsa_groups",
        _torch_qsa_compress,
        qwen4_vendor_compress_qsa_groups,
        _qsa_compress_input,
    )


def _make_cos_sin_cache(device, dtype):
    positions = torch.arange(SEQUENCE_LENGTH, device=device, dtype=torch.float32)
    frequencies = 1.0 / (
        1_000_000
        ** (
            torch.arange(0, ROTARY_DIM, 2, device=device, dtype=torch.float32)
            / ROTARY_DIM
        )
    )
    angles = positions[:, None] * frequencies[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)


def _qsa_fused_compress_input(shape, dtype, device):
    (rows,) = shape
    torch.manual_seed(8000 + rows)
    raw, table, requests, positions, slots, rope = _make_qsa_compress_inputs(
        rows, dtype, device
    )
    compressed_cache = torch.zeros(
        (rows * COMPRESSED_PAGES_PER_REQUEST, PAGE_SIZE, 1, QSA_DIM),
        device=device,
        dtype=dtype,
    )
    weight = torch.randn((QSA_DIM,), device=device, dtype=dtype) * 0.01
    cos_sin = _make_cos_sin_cache(device, dtype)
    yield (
        raw,
        table,
        requests,
        positions,
        slots,
        compressed_cache,
        weight,
        cos_sin,
        QSA_RATIO,
        EPS,
        ROTARY_DIM,
        MROPE_SECTION,
        True,
        rope,
    )


def _torch_qsa_fused_compress(
    raw_cache,
    block_table,
    requests,
    positions,
    compressed_slots,
    compressed_cache,
    weight,
    cos_sin,
    ratio,
    eps,
    rotary_dim,
    mrope_section,
    mrope_interleaved,
    rope_cache,
):
    pooled, axes = _torch_qsa_compress(
        raw_cache,
        block_table,
        requests,
        positions,
        compressed_slots,
        ratio,
        rope_cache,
    )
    pooled_fp32 = pooled[:, 0].float()
    normalized = (
        pooled_fp32
        * torch.rsqrt(pooled_fp32.square().mean(-1, keepdim=True) + eps)
        * (weight.float() + 1.0)
    ).to(raw_cache.dtype)
    frequency = torch.arange(rotary_dim // 2, device=raw_cache.device)
    if mrope_interleaved:
        use_height = (frequency % 3 == 1) & (frequency < 3 * mrope_section[1])
        use_width = (frequency % 3 == 2) & (frequency < 3 * mrope_section[2])
    else:
        height_start = mrope_section[0]
        width_start = height_start + mrope_section[1]
        use_height = (frequency >= height_start) & (frequency < width_start)
        use_width = (frequency >= width_start) & (
            frequency < width_start + mrope_section[2]
        )
    rope_positions = torch.where(
        use_height[None, :],
        axes[:, 1, None],
        torch.where(use_width[None, :], axes[:, 2, None], axes[:, 0, None]),
    )
    cos = cos_sin[rope_positions, frequency]
    sin = cos_sin[rope_positions, frequency + rotary_dim // 2]
    first = normalized[:, : rotary_dim // 2]
    second = normalized[:, rotary_dim // 2 : rotary_dim]
    stored = normalized.clone()
    stored[:, : rotary_dim // 2] = (first * cos - second * sin).to(raw_cache.dtype)
    stored[:, rotary_dim // 2 : rotary_dim] = (second * cos + first * sin).to(
        raw_cache.dtype
    )
    compressed_cache.view(-1, QSA_DIM).index_copy_(0, compressed_slots, stored)


@pytest.mark.qwen4_compress_norm_mrope_store_groups
def test_qwen4_compress_norm_mrope_store_groups():
    _benchmark(
        "qwen4_compress_norm_mrope_store_groups",
        _torch_qsa_fused_compress,
        qwen4_compress_norm_mrope_store_groups,
        _qsa_fused_compress_input,
    )


def _ple_gather_input(shape, dtype, device):
    (rows,) = shape
    torch.manual_seed(9000 + rows)
    storage = torch.randn((8192, 65, 3), device=device, dtype=dtype)
    state = storage.transpose(1, 2)
    indices = torch.arange(1, rows + 1, device=device, dtype=torch.int64)
    yield state, indices


def _torch_ple_gather(state, indices):
    return torch.index_select(state, 0, indices)


@pytest.mark.qwen4_ple_state_gather
def test_qwen4_ple_state_gather():
    _benchmark(
        "qwen4_ple_state_gather",
        _torch_ple_gather,
        ple_state_gather,
        _ple_gather_input,
    )


def _ple_scatter_input(shape, dtype, device):
    (rows,) = shape
    torch.manual_seed(10000 + rows)
    storage = torch.randn((8192, 65, 3), device=device, dtype=dtype)
    state = storage.transpose(1, 2)
    indices = torch.arange(1, rows + 1, device=device, dtype=torch.int64)
    values = torch.randn((rows, 3, 65), device=device, dtype=dtype)
    mask = torch.ones((rows,), device=device, dtype=torch.bool)
    yield state, indices, values, mask


def _torch_ple_scatter(state, indices, values, mask):
    _ = mask
    state.index_copy_(0, indices, values)


def _gems_ple_scatter(state, indices, values, mask):
    ple_state_scatter_(state, indices, values, write_mask=mask)


@pytest.mark.qwen4_ple_state_scatter
def test_qwen4_ple_state_scatter():
    _benchmark(
        "qwen4_ple_state_scatter",
        _torch_ple_scatter,
        _gems_ple_scatter,
        _ple_scatter_input,
    )


pytestmark = pytest.mark.skipif(
    flaggems_vllm.device == "cpu" or not torch.cuda.is_available(),
    reason="Qwen4 Triton benchmarks require a CUDA accelerator",
)
