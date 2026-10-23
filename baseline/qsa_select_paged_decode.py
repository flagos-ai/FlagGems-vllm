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

"""qsa_select_paged_decode baseline（方案 B）——QSA decode 阶段压缩块打分+选择。

native：vllm.models.qwen4_exp.nvidia.ops.qsa_indexer.qsa_select_paged_decode
    （src: qsa_indexer.py#L481）。签名（已回源核对）：
        qsa_select_paged_decode(q, k_cache, page_table, visible_blocks,
            token_topk, compress_ratio, decode_query_len, block_indices) -> None
    结果写入 block_indices 输出缓冲（原地）。约束（源码 assert）：
        token_topk % compress_ratio == 0；
        block_indices.shape == (q.shape[0], token_topk // compress_ratio)；
        q.shape[0] % decode_query_len == 0；
        page_table.shape[0] == q.shape[0] // decode_query_len；
        visible_blocks.shape == (q.shape[0],)。

shape 为 vllm 源码推断，非 FlagGems-vllm 基准（CSV「未找到对应benchmark」）。
    维度取自 QSAIndexer 配置与调用点（nvidia/indexer_qsa.py, tests/models/qwen4_exp）：
        head_dim=128, num_kv_heads=1, index_n_heads=64,
        token_topk=indexer_budget=2048, compress_ratio=4, page_size=64。
    q: [num_rows, heads, head_dim]；k_cache: [blocks, page_size, 1, head_dim]。
    decode: num_rows = num_requests * decode_query_len。
"""

import importlib

import torch

OP_NAME = "qsa_select_paged_decode"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 结果写入 block_indices 输出缓冲

HEAD_DIM = 128
NUM_HEADS = 64
NUM_KV_HEADS = 1
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
PAGE_SIZE = 64
# (num_requests, decode_query_len, max_pages) —— 解码常见档；上下文越长 max_pages 越大
_CASES = [
    (16, 1, 32),
    (64, 1, 64),
    (128, 1, 128),
    (256, 1, 256),
    (32, 2, 128),
]


def native():
    try:
        mod = importlib.import_module(
            "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer")
    except ImportError:
        return None
    op = getattr(mod, "qsa_select_paged_decode", None)
    return op if callable(op) else None


def grid():
    return [
        {"num_requests": r, "decode_query_len": d, "max_pages": p}
        for (r, d, p) in _CASES
    ]


def build_inputs(binding, dtype, device):
    r = binding["num_requests"]
    dq = binding["decode_query_len"]
    max_pages = binding["max_pages"]
    num_rows = r * dq

    q = torch.randn(num_rows, NUM_HEADS, HEAD_DIM, dtype=dtype, device=device)
    blocks = r * max_pages
    k_cache = torch.randn(blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                          dtype=dtype, device=device)
    page_table = torch.zeros(r, max_pages, dtype=torch.int32, device=device)
    # visible_blocks：每 query 可见压缩块数（不超过 page_table 覆盖列数/压缩比）
    max_visible = max_pages * PAGE_SIZE // COMPRESS_RATIO
    visible_blocks = torch.full((num_rows,), min(max_visible, TOKEN_TOPK // COMPRESS_RATIO),
                                dtype=torch.int32, device=device)
    block_indices = torch.empty(num_rows, TOKEN_TOPK // COMPRESS_RATIO,
                                dtype=torch.int32, device=device)

    args = (q, k_cache, page_table, visible_blocks,
            TOKEN_TOPK, COMPRESS_RATIO, dq, block_indices)
    return args, {}


def key_shape(binding):
    return [binding["num_requests"], binding["decode_query_len"],
            binding["max_pages"], NUM_HEADS, HEAD_DIM]


def config(binding, dtype):
    r = binding["num_requests"]
    dq = binding["decode_query_len"]
    max_pages = binding["max_pages"]
    num_rows = r * dq
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [num_rows, NUM_HEADS, HEAD_DIM], "dtype": dt},
            "k_cache": {"shape": [r * max_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM],
                        "dtype": dt},
            "page_table": {"shape": [r, max_pages], "dtype": "torch.int32"},
            "visible_blocks": {"shape": [num_rows], "dtype": "torch.int32"},
        },
        "outputs": {"block_indices": {"shape": [num_rows, TOKEN_TOPK // COMPRESS_RATIO],
                                      "dtype": "torch.int32"}},
        "dims": {"num_requests": r, "decode_query_len": dq, "max_pages": max_pages,
                 "heads": NUM_HEADS, "head_dim": HEAD_DIM, "page_size": PAGE_SIZE,
                 "token_topk": TOKEN_TOPK, "compress_ratio": COMPRESS_RATIO,
                 "shape_source": "vllm 源码推断（无 benchmark）"},
    }
