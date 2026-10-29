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
    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        qsa_select_paged_decode as vllm_qsa_select_paged_decode,
    )

    _HAS_VLLM_QSA = True
except Exception:
    vllm_qsa_select_paged_decode = None
    _HAS_VLLM_QSA = False


# --- Fixed production constants (Qwen3.8-Flash-Next-FP8-p32768d1024) ---
HEAD_DIM = 128
NUM_HEADS = 4
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
DECODE_QUERY_LEN = 1
PAGE_SIZE = 196
MAX_PAGES = 88
MIN_VISIBLE = 8192  # = 32768 / compress_ratio
COLUMNS = MAX_PAGES * PAGE_SIZE  # = 17248
NUM_BLOCKS_SHARED = 2712
NUM_BLOCKS_SMALL = 512

# Representative subset of the Qwen3.8-Flash-Next-FP8-p32768d1024 shape trace.
# The trace contains 149 unique shapes; every one shares page_size=196,
# max_pages=88 (=> columns = 17248), heads=4, head_dim=128, token_topk=2048,
# compress_ratio=4 (=> block_topk=512) and decode_query_len=1.  Only
# (num_requests, num_blocks) varies, over two physical block-pool layouts:
# a shared pool of 2712 blocks and a small pool of 512 blocks.
QSA_DECODE_SCENARIOS_FP8 = [
    # --- 2712-block pool: row-count sweep ---
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
    # --- 512-block pool: small-pool layout ---
    (32, NUM_BLOCKS_SMALL),
    (128, NUM_BLOCKS_SMALL),
    (256, NUM_BLOCKS_SMALL),
    (512, NUM_BLOCKS_SMALL),
]


class QsaSelectPagedDecodeBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = QSA_DECODE_SCENARIOS_FP8
        self.shape_desc = "num_requests, num_blocks"

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield self.qsa_input_fn(config, dtype, self.device)

    def qsa_input_fn(self, config, dtype, device):
        num_requests, num_blocks = config
        rows = num_requests * DECODE_QUERY_LEN
        block_topk = TOKEN_TOPK // COMPRESS_RATIO

        q = torch.randn(
            rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
        ).to(dtype)
        k_cache = torch.randn(
            num_blocks,
            PAGE_SIZE,
            1,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        ).to(dtype)

        flat = torch.randperm(num_blocks, device=device)
        entry_ids = torch.arange(num_requests * MAX_PAGES, device=device)
        page_table = (
            flat[entry_ids % num_blocks].view(num_requests, MAX_PAGES).to(torch.int32)
        )
        visible_blocks = torch.randint(
            MIN_VISIBLE,
            COLUMNS + 1,
            (rows,),
            device=device,
            dtype=torch.int32,
        )
        block_indices = torch.empty(
            (rows, block_topk), dtype=torch.int32, device=device
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
    reason="requires the vllm baseline",
)
def test_qsa_select_paged_decode_perf_fp8():
    bench = QsaSelectPagedDecodeBenchmark(
        op_name="qsa_select_paged_decode",
        torch_op=vllm_qsa_select_paged_decode,
        dtypes=[torch.float8_e4m3fn],
    )
    bench.set_gems(flaggems_vllm.qsa_select_paged_decode)
    bench.run()
