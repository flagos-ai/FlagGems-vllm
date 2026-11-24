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

import pytest
import torch

import flaggems_vllm

from . import base

try:
    if flaggems_vllm.vendor_name == "metax":
        from vllm_metax.models.qwen4_exp.metax.ops.qsa_indexer_fallback import (
            qsa_select_paged_prefill as vendor_qsa,
        )
    else:
        from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
            qsa_select_paged_prefill as vendor_qsa,
        )

    _HAS_VENDOR = True
    _VENDOR_ERROR = ""
except Exception as exc:
    vendor_qsa = None
    _HAS_VENDOR = False
    _VENDOR_ERROR = f"{type(exc).__name__}: {exc}"


HEAD_DIM = 128
NUM_HEADS = 4
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO
PAGE_SIZE = 196

# name, query lengths, maximum sequence length, physical pages, table columns
QSA_PREFILL_SCENARIOS = [
    ("first_chunk", (16384,), 16384, 2712, 88),
    ("budget_exact", (16384,), 32768, 2712, 88),
    ("dominant_long", (16331,), 33842, 2712, 88),
    ("tail112", (16368,), 32853, 2712, 88),
    ("tail113", (16369,), 32851, 2712, 88),
    ("two_balanced", (10371, 5958), 32932, 2712, 88),
    ("two_skewed", (16330, 54), 32822, 2712, 88),
    ("short_tail", (2316,), 33843, 2712, 88),
    ("compact", (3200,) + (112,) * 38 + (96,) * 15, 4100, None, None),
    ("long100k", (16384,), 118784, None, None),
]


def _to_dtype(value, dtype):
    if dtype == torch.float8_e4m3fn and flaggems_vllm.vendor_name == "metax":
        result = torch.empty(value.shape, dtype=dtype, device=value.device)
        result.copy_(value.cpu().to(dtype))
        return result
    return value.to(dtype)


def make_inputs(scenario, dtype, device, seed=20261121):
    _, query_lengths, max_seq_len, physical_pages, table_columns = scenario
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = sum(query_lengths)
    compressed_lengths = [
        (max_seq_len + COMPRESS_RATIO - 1) // COMPRESS_RATIO
        for _ in query_lengths
    ]
    pages_per_request = [
        (length + PAGE_SIZE - 1) // PAGE_SIZE for length in compressed_lengths
    ]
    if physical_pages is None:
        physical_pages = sum(pages_per_request)
    if table_columns is None:
        table_columns = max(pages_per_request)

    q = torch.randn(
        rows,
        NUM_HEADS,
        HEAD_DIM,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        physical_pages,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    q = _to_dtype(q, dtype)
    k_cache = _to_dtype(k_cache, dtype)

    page_order = torch.randperm(physical_pages, generator=generator, device=device)
    page_table = torch.full(
        (len(query_lengths), table_columns), -1, dtype=torch.int32, device=device
    )
    cursor = 0
    for request, count in enumerate(pages_per_request):
        page_table[request, :count] = page_order[cursor : cursor + count]
        cursor += count

    starts = [0]
    visible_parts = []
    for query_len in query_lengths:
        starts.append(starts[-1] + query_len)
        prefix_len = max_seq_len - query_len
        visible_parts.append(
            torch.arange(
                prefix_len + 1,
                prefix_len + query_len + 1,
                dtype=torch.int32,
                device=device,
            )
            // COMPRESS_RATIO
        )
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device=device)
    visible_blocks = torch.cat(visible_parts)
    block_indices = torch.empty(
        (rows, BLOCK_TOPK), dtype=torch.int32, device=device
    )
    return (
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        TOKEN_TOPK,
        COMPRESS_RATIO,
        max(query_lengths),
        block_indices,
        max_seq_len,
    )


class QsaSelectPagedPrefillBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = QSA_PREFILL_SCENARIOS
        self.shape_desc = (
            "name, query lengths, maximum sequence length, physical pages, "
            "table columns (H=4, D=128, page=196, block_topk=512)"
        )

    def get_input_iter(self, dtype):
        for scenario in self.shapes:
            yield make_inputs(scenario, dtype, self.device)


@pytest.mark.qsa_select_paged_prefill
@pytest.mark.skipif(
    not _HAS_VENDOR,
    reason=f"requires the vendor vLLM baseline: {_VENDOR_ERROR}".rstrip(": "),
)
def test_qsa_select_paged_prefill_perf_bf16():
    bench = QsaSelectPagedPrefillBenchmark(
        op_name="qsa_select_paged_prefill",
        torch_op=vendor_qsa,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flaggems_vllm.qsa_select_paged_prefill)
    bench.run()


@pytest.mark.qsa_select_paged_prefill
@pytest.mark.skipif(
    not _HAS_VENDOR,
    reason=f"requires the vendor vLLM baseline: {_VENDOR_ERROR}".rstrip(": "),
)
def test_qsa_select_paged_prefill_perf_fp8():
    bench = QsaSelectPagedPrefillBenchmark(
        op_name="qsa_select_paged_prefill",
        torch_op=vendor_qsa,
        dtypes=[torch.float8_e4m3fn],
    )
    bench.set_gems(flaggems_vllm.qsa_select_paged_prefill)
    bench.run()
