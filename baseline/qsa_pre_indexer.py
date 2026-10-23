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

"""qsa_pre_indexer baseline（方案 B）——QSA 索引前处理：Q 归一化 + K 压缩 + 循环状态更新。

native：vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer.qsa_pre_indexer
    （src: qsa_pre_indexer.py#L403）。签名（已回源核对）：
        qsa_pre_indexer(q, k, positions, cos_sin_cache, q_norm_weight,
            k_norm_weight, eps, q_out, state_cache, state_slots,
            state_block_table, query_start_loc, logical_positions,
            compressed_cache, compressed_slots, k_work_metadata, *,
            compress_ratio, mrope_section=None, rope_pos_offset=None) -> None
    结果写入 q_out / state_cache / compressed_cache（原地）。

shape 为 vllm 源码推断，非 FlagGems-vllm 基准（CSV「未找到对应benchmark」）。
    张量形状取自 kernel 内 assert（qsa_pre_indexer.py#L426 起）与调用点
    （nvidia/indexer_qsa.py#L262）：
        q: [T, num_q_heads*head_dim]；k: [T, head_dim]；q_out: [T, num_q_heads, head_dim]；
        cos_sin_cache: [max_pos, head_dim//2]；state/compressed_cache: [blocks, page, 1, head];
        k_work_metadata: [num_k_work, 2]。
    维度取自 QSAIndexer 配置：index_n_heads=64, index_head_dim(head_dim)=128,
        rotary_dim=64, num_kv_heads=1, compress_ratio=4。走非 mrope、非 2D positions、
        rope_pos_offset=None 的基本路径。
    注意：该算子依赖循环状态缓存与 k_work_metadata 的真实调度内容；此处按 shape
    契约构造占位输入用于 baseline 采集，实机若需精确调度需由 QSAMetadataBuilder 提供。
"""

import importlib

import torch

OP_NAME = "qsa_pre_indexer"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 写入 q_out / state_cache / compressed_cache

NUM_Q_HEADS = 64
HEAD_DIM = 128
NUM_KV_HEADS = 1
COMPRESS_RATIO = 4
STATE_SIZE = 64        # 循环状态缓存 page 大小（block_size 档）
COMP_PAGE_SIZE = 64    # 压缩缓存 page 大小
EPS = 1e-6
# (num_requests, seqlen_per_req, max_pages)
_CASES = [
    (1, 1024, 32),
    (2, 512, 16),
    (4, 256, 8),
    (1, 4096, 128),
]


def native():
    try:
        mod = importlib.import_module(
            "vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer")
    except ImportError:
        return None
    op = getattr(mod, "qsa_pre_indexer", None)
    return op if callable(op) else None


def grid():
    return [{"num_requests": r, "seqlen": s, "max_pages": p} for (r, s, p) in _CASES]


def build_inputs(binding, dtype, device):
    r = binding["num_requests"]
    seqlen = binding["seqlen"]
    max_pages = binding["max_pages"]
    T = r * seqlen

    q = torch.randn(T, NUM_Q_HEADS * HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(T, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(T, dtype=torch.int64, device=device) % (seqlen)
    cos_sin_cache = torch.randn(seqlen + 1, HEAD_DIM // 2, dtype=dtype, device=device)
    q_norm_weight = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    k_norm_weight = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    q_out = torch.empty(T, NUM_Q_HEADS, HEAD_DIM, dtype=dtype, device=device)

    state_blocks = r * max_pages
    state_cache = torch.zeros(state_blocks, STATE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                              dtype=dtype, device=device)
    comp_blocks = r * max_pages
    compressed_cache = torch.zeros(comp_blocks, COMP_PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                                   dtype=dtype, device=device)
    state_slots = torch.arange(T, dtype=torch.int32, device=device)
    compressed_slots = (torch.arange(T, dtype=torch.int32, device=device)
                        // COMPRESS_RATIO)
    state_block_table = torch.zeros(r, max_pages, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(0, T + 1, seqlen, dtype=torch.int32, device=device)
    logical_positions = torch.arange(T, dtype=torch.int64, device=device) % seqlen
    # k_work_metadata: [num_k_work, 2]（每工作块 [start, end)）；占位 1 个空工作块
    k_work_metadata = torch.zeros(1, 2, dtype=torch.int32, device=device)

    args = (q, k, positions, cos_sin_cache, q_norm_weight, k_norm_weight, EPS,
            q_out, state_cache, state_slots, state_block_table, query_start_loc,
            logical_positions, compressed_cache, compressed_slots, k_work_metadata)
    kwargs = {"compress_ratio": COMPRESS_RATIO,
              "mrope_section": None, "rope_pos_offset": None}
    return args, kwargs


def key_shape(binding):
    return [binding["num_requests"], binding["seqlen"],
            binding["max_pages"], NUM_Q_HEADS, HEAD_DIM]


def config(binding, dtype):
    r = binding["num_requests"]
    seqlen = binding["seqlen"]
    max_pages = binding["max_pages"]
    T = r * seqlen
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [T, NUM_Q_HEADS * HEAD_DIM], "dtype": dt},
            "k": {"shape": [T, HEAD_DIM], "dtype": dt},
            "positions": {"shape": [T], "dtype": "torch.int64"},
            "cos_sin_cache": {"shape": [seqlen + 1, HEAD_DIM // 2], "dtype": dt},
            "state_cache": {"shape": [r * max_pages, STATE_SIZE, NUM_KV_HEADS, HEAD_DIM],
                            "dtype": dt},
            "compressed_cache": {"shape": [r * max_pages, COMP_PAGE_SIZE,
                                           NUM_KV_HEADS, HEAD_DIM], "dtype": dt},
        },
        "outputs": {"q_out": {"shape": [T, NUM_Q_HEADS, HEAD_DIM], "dtype": dt}},
        "dims": {"num_requests": r, "seqlen": seqlen, "max_pages": max_pages,
                 "q_heads": NUM_Q_HEADS, "head_dim": HEAD_DIM,
                 "compress_ratio": COMPRESS_RATIO,
                 "shape_source": "vllm 源码推断（无 benchmark）"},
    }
