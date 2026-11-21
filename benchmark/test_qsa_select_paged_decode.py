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

from unittest.mock import patch

import pytest
import torch

import flaggems_vllm

from . import base


def _hygon_qsa_select_paged_decode(
    q,
    k_cache,
    page_table,
    visible_blocks,
    token_topk,
    compress_ratio,
    decode_query_len,
    block_indices,
):
    rows = q.shape[0]
    requests = rows // decode_query_len
    token_to_req = torch.arange(
        requests, device=q.device, dtype=torch.int32
    ).repeat_interleave(decode_query_len)
    sequence_lengths = (
        visible_blocks.view(requests, decode_query_len).max(dim=1).values
        * compress_ratio
    ).to(torch.int32)
    query_positions = visible_blocks * compress_ratio - 1
    logits, actual_visible = qsa_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
    )
    torch.ops._C.top_k_per_row_decode(
        logits,
        1,
        actual_visible,
        block_indices,
        rows,
        logits.stride(0),
        logits.stride(1),
        token_topk // compress_ratio,
    )


def _ppu_device_capability(cls, device_id=0):
    from vllm.platforms.interface import DeviceCapability

    major, minor = torch.cuda.get_device_capability(device_id)
    return DeviceCapability(major=major, minor=minor)


try:
    if flaggems_vllm.vendor_name == "metax":
        from vllm_metax.models.qwen4_exp.metax.ops.qsa_indexer_fallback import (
            qsa_select_paged_decode as vllm_qsa_select_paged_decode,
        )
    elif flaggems_vllm.vendor_name == "hygon":
        from vllm.models.qwen4_exp.amd.ops.qsa import qsa_mqa_paged
        from vllm.platforms import current_platform

        current_platform.import_kernels()
        vllm_qsa_select_paged_decode = _hygon_qsa_select_paged_decode
    else:
        from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
            qsa_select_paged_decode as vllm_qsa_select_paged_decode,
        )

    if flaggems_vllm.vendor_name == "thead":
        from vllm.platforms import current_platform

        assert current_platform.is_ppu(), "run PPU benchmarks with VLLM_PLUGINS=ppu"
        current_platform.import_kernels()
    _HAS_VLLM_QSA = True
    _VLLM_QSA_ERROR = ""
except Exception as exc:
    vllm_qsa_select_paged_decode = None
    _HAS_VLLM_QSA = False
    _VLLM_QSA_ERROR = f"{type(exc).__name__}: {exc}"


@pytest.fixture
def _scoped_ppu_device_capability():
    if flaggems_vllm.vendor_name != "thead":
        yield
        return

    from vllm.platforms import current_platform

    with patch.object(
        type(current_platform),
        "get_device_capability",
        classmethod(_ppu_device_capability),
    ):
        yield


HEAD_DIM = 128
NUM_HEADS = 4
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO
DECODE_QUERY_LEN = 1
PAGE_SIZE = 196
MAX_PAGES = 88
MIN_VISIBLE = 8192
COLUMNS = MAX_PAGES * PAGE_SIZE
NUM_BLOCKS_SHARED = 2712
NUM_BLOCKS_SMALL = 512
assert (
    COLUMNS > BLOCK_TOPK
), f"top-k degenerates into full selection: {COLUMNS=} <= {BLOCK_TOPK=}"


QSA_DECODE_SCENARIOS = [
    (1, NUM_BLOCKS_SHARED),
    (8, NUM_BLOCKS_SHARED),
    (32, NUM_BLOCKS_SHARED),
    (53, NUM_BLOCKS_SHARED),
    (56, NUM_BLOCKS_SHARED),
    (64, NUM_BLOCKS_SHARED),
    (128, NUM_BLOCKS_SHARED),
    (256, NUM_BLOCKS_SHARED),
    (320, NUM_BLOCKS_SHARED),
    (512, NUM_BLOCKS_SHARED),
    (32, NUM_BLOCKS_SMALL),
    (128, NUM_BLOCKS_SMALL),
    (256, NUM_BLOCKS_SMALL),
    (512, NUM_BLOCKS_SMALL),
]


def _to_dtype(value, dtype):
    if dtype == torch.float8_e4m3fn and flaggems_vllm.vendor_name == "metax":
        result = torch.empty(value.shape, dtype=dtype, device=value.device)
        result.copy_(value.cpu().to(dtype))
        return result
    return value.to(dtype)


class QsaSelectPagedDecodeBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = QSA_DECODE_SCENARIOS
        self.shape_desc = "num_requests, num_blocks (88x196 columns, block_topk=512)"

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield self.qsa_input_fn(config, dtype, self.device)

    def qsa_input_fn(self, config, dtype, device, seed=0):
        num_requests, num_blocks = config
        rows = num_requests * DECODE_QUERY_LEN
        g = torch.Generator(device=device).manual_seed(seed)

        q = torch.randn(
            rows,
            NUM_HEADS,
            HEAD_DIM,
            generator=g,
            device=device,
            dtype=torch.bfloat16,
        )
        k_cache = torch.randn(
            num_blocks,
            PAGE_SIZE,
            1,
            HEAD_DIM,
            generator=g,
            device=device,
            dtype=torch.bfloat16,
        )
        q = _to_dtype(q, dtype)
        k_cache = _to_dtype(k_cache, dtype)

        flat = torch.randperm(num_blocks, generator=g, device=device)
        entry_ids = torch.arange(num_requests * MAX_PAGES, device=device)
        page_table = (
            flat[entry_ids % num_blocks].view(num_requests, MAX_PAGES).to(torch.int32)
        )
        visible_blocks = torch.randint(
            MIN_VISIBLE,
            COLUMNS + 1,
            (rows,),
            generator=g,
            device=device,
            dtype=torch.int32,
        )
        block_indices = torch.empty(
            (rows, BLOCK_TOPK), dtype=torch.int32, device=device
        )
        return (
            q,
            k_cache,
            page_table,
            visible_blocks,
            TOKEN_TOPK,
            COMPRESS_RATIO,
            DECODE_QUERY_LEN,
            block_indices,
        )


@pytest.mark.qsa_select_paged_decode
@pytest.mark.skipif(
    not _HAS_VLLM_QSA,
    reason=f"requires the vendor vLLM baseline: {_VLLM_QSA_ERROR}".rstrip(": "),
)
def test_qsa_select_paged_decode_perf_bf16(_scoped_ppu_device_capability):
    bench = QsaSelectPagedDecodeBenchmark(
        op_name="qsa_select_paged_decode",
        torch_op=vllm_qsa_select_paged_decode,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flaggems_vllm.qsa_select_paged_decode)
    bench.run()


@pytest.mark.qsa_select_paged_decode
@pytest.mark.skipif(
    not _HAS_VLLM_QSA,
    reason=f"requires the vendor vLLM baseline: {_VLLM_QSA_ERROR}".rstrip(": "),
)
def test_qsa_select_paged_decode_perf_fp8(_scoped_ppu_device_capability):
    bench = QsaSelectPagedDecodeBenchmark(
        op_name="qsa_select_paged_decode",
        torch_op=vllm_qsa_select_paged_decode,
        dtypes=[torch.float8_e4m3fn],
    )
    bench.set_gems(flaggems_vllm.qsa_select_paged_decode)
    bench.run()
