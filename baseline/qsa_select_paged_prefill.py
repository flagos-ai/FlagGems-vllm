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

"""qsa_select_paged_prefill baseline（方案 B）——QSA prefill 阶段压缩块打分+选择。

native：vllm.models.qwen4_exp.nvidia.ops.qsa_indexer.qsa_select_paged_prefill
    （src: qsa_indexer.py#L550）。签名（已回源核对）：
        qsa_select_paged_prefill(q, k_cache, page_table, query_start_loc,
            visible_blocks, token_topk, compress_ratio, max_query_len,
            block_indices) -> None
    结果写入 block_indices 输出缓冲（原地）。约束（源码 assert）：
        token_topk % compress_ratio == 0；
        block_indices.shape == (q.shape[0], token_topk // compress_ratio)。
    内部按 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB 分块处理。

shape 为 vllm 源码推断，非 FlagGems-vllm 基准（CSV「未找到对应benchmark」）。
    维度取自 QSAIndexer 配置与调用点：
        head_dim=128, num_kv_heads=1, index_n_heads=64,
        token_topk=2048, compress_ratio=4, page_size=64。
    q: [num_tokens, heads, head_dim]（packed prefill）；
    page_table: [num_requests, max_pages]；query_start_loc: [num_requests+1] int32。
"""

import importlib

import torch

OP_NAME = "qsa_select_paged_prefill"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 结果写入 block_indices 输出缓冲

HEAD_DIM = 128
NUM_HEADS = 64
NUM_KV_HEADS = 1
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
PAGE_SIZE = 64
# (num_requests, seqlen_per_req, max_pages) —— prefill 常见档
_CASES = [
    (1, 4096, 64),
    (2, 2048, 32),
    (4, 1024, 16),
    (1, 8192, 128),
]


def native():
    try:
        mod = importlib.import_module(
            "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer")
    except ImportError:
        return None
    op = getattr(mod, "qsa_select_paged_prefill", None)
    return op if callable(op) else None


def grid():
    return [
        {"num_requests": r, "seqlen": s, "max_pages": p}
        for (r, s, p) in _CASES
    ]


def build_inputs(binding, dtype, device):
    r = binding["num_requests"]
    seqlen = binding["seqlen"]
    max_pages = binding["max_pages"]
    num_tokens = r * seqlen

    q = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=dtype, device=device)
    blocks = r * max_pages
    k_cache = torch.randn(blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                          dtype=dtype, device=device)
    page_table = torch.zeros(r, max_pages, dtype=torch.int32, device=device)
    # query_start_loc：packed prefill 每请求起始偏移 + 终止偏移
    query_start_loc = torch.arange(0, num_tokens + 1, seqlen,
                                   dtype=torch.int32, device=device)
    max_visible = max_pages * PAGE_SIZE // COMPRESS_RATIO
    visible_blocks = torch.full((num_tokens,),
                                min(max_visible, TOKEN_TOPK // COMPRESS_RATIO),
                                dtype=torch.int32, device=device)
    block_indices = torch.empty(num_tokens, TOKEN_TOPK // COMPRESS_RATIO,
                                dtype=torch.int32, device=device)

    args = (q, k_cache, page_table, query_start_loc, visible_blocks,
            TOKEN_TOPK, COMPRESS_RATIO, seqlen, block_indices)
    return args, {}


def key_shape(binding):
    return [binding["num_requests"], binding["seqlen"],
            binding["max_pages"], NUM_HEADS, HEAD_DIM]


def config(binding, dtype):
    r = binding["num_requests"]
    seqlen = binding["seqlen"]
    max_pages = binding["max_pages"]
    num_tokens = r * seqlen
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [num_tokens, NUM_HEADS, HEAD_DIM], "dtype": dt},
            "k_cache": {"shape": [r * max_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM],
                        "dtype": dt},
            "page_table": {"shape": [r, max_pages], "dtype": "torch.int32"},
            "query_start_loc": {"shape": [r + 1], "dtype": "torch.int32"},
            "visible_blocks": {"shape": [num_tokens], "dtype": "torch.int32"},
        },
        "outputs": {"block_indices": {"shape": [num_tokens, TOKEN_TOPK // COMPRESS_RATIO],
                                      "dtype": "torch.int32"}},
        "dims": {"num_requests": r, "seqlen": seqlen, "max_pages": max_pages,
                 "heads": NUM_HEADS, "head_dim": HEAD_DIM, "page_size": PAGE_SIZE,
                 "token_topk": TOKEN_TOPK, "compress_ratio": COMPRESS_RATIO,
                 "shape_source": "vllm 源码推断（无 benchmark）"},
    }
