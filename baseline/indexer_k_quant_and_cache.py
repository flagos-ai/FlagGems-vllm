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

"""indexer_k_quant_and_cache baseline（DSA indexer K 的 FP8 量化 + 写 paged cache）。

native：vllm._custom_ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping,
        quant_block_size, kv_cache_dtype) -> ()。原地把量化后的 K 与 scale 写进
        kv_cache。
    Python 封装见 vllm/_custom_ops.py#L3121（内部转 torch.ops._C_cache_ops.
    indexer_k_quant_and_cache）；STABLE_TORCH_LIBRARY 绑定签名见
    csrc/libtorch_stable/torch_bindings.cpp#L1009。

输入构造复刻 FlagGems-vllm/benchmark/test_indexer_k_quant_and_cache.py 的
IndexerKQuantAndCacheBenchmark.get_input_iter：
    k            = randn(num_tokens, head_dim) dtype
    slot_mapping = randperm(num_blocks*block_size)[:num_tokens].long()
    cache_stride = head_dim + head_dim*4 // quant_block_size
    kv_cache     = empty(num_blocks, block_size, cache_stride) uint8
    quant_block_size 标量；kv_cache_dtype 传 "ue8m0"（benchmark 里 scale_fmt）。

    注：benchmark 走 vllm 封装时把 scale_fmt 作为第 5 个位置参数 kv_cache_dtype
    传入（见 test 文件 vllm_indexer），故这里 kv_cache_dtype="ue8m0"。

shape 来自 benchmark set_shapes：
    (num_tokens, num_blocks, block_size, head_dim, quant_block_size) =
    (128,16,16,128,128)/(512,64,16,128,128)/(1024,128,16,512,128)/
    (2048,256,16,512,128)。DTYPES 复刻 benchmark 的 [fp16, bf16]。
"""

import importlib

import torch

OP_NAME = "indexer_k_quant_and_cache"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True  # kv_cache 原地写

_KV_CACHE_DTYPE = "ue8m0"

# benchmark set_shapes：(num_tokens, num_blocks, block_size, head_dim, quant_block_size)
_SHAPES = [
    (128, 16, 16, 128, 128),
    (512, 64, 16, 128, 128),
    (1024, 128, 16, 512, 128),
    (2048, 256, 16, 512, 128),
]


def native():
    """解析 vllm._custom_ops.indexer_k_quant_and_cache；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "indexer_k_quant_and_cache", None)
    return op if callable(op) else None


def grid():
    return [{"num_tokens": nt, "num_blocks": nb, "block_size": bs,
             "head_dim": hd, "quant_block_size": qbs}
            for (nt, nb, bs, hd, qbs) in _SHAPES]


def build_inputs(binding, dtype, device):
    nt = binding["num_tokens"]
    nb = binding["num_blocks"]
    bs = binding["block_size"]
    hd = binding["head_dim"]
    qbs = binding["quant_block_size"]
    k = torch.randn(nt, hd, dtype=dtype, device=device)
    slot_mapping = torch.randperm(nb * bs, device=device)[:nt].to(torch.int64)
    cache_stride = hd + hd * 4 // qbs
    kv_cache = torch.empty(nb, bs, cache_stride, dtype=torch.uint8, device=device)
    return (k, kv_cache, slot_mapping, qbs, _KV_CACHE_DTYPE), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["num_blocks"],
            binding["block_size"], binding["head_dim"],
            binding["quant_block_size"]]


def config(binding, dtype):
    nt = binding["num_tokens"]
    nb = binding["num_blocks"]
    bs = binding["block_size"]
    hd = binding["head_dim"]
    qbs = binding["quant_block_size"]
    cache_stride = hd + hd * 4 // qbs
    return {
        "inputs": {
            "k": {"shape": [nt, hd], "dtype": repr(dtype)},
            "kv_cache": {"shape": [nb, bs, cache_stride], "dtype": "torch.uint8",
                         "note": "原地写回量化 K + scale"},
            "slot_mapping": {"shape": [nt], "dtype": "torch.int64"},
            "quant_block_size": {"scalar": qbs},
            "kv_cache_dtype": {"scalar": _KV_CACHE_DTYPE},
        },
        "outputs": {
            "kv_cache": {"shape": [nb, bs, cache_stride], "dtype": "torch.uint8"},
        },
        "dims": {"num_tokens": nt, "num_blocks": nb, "block_size": bs,
                 "head_dim": hd, "quant_block_size": qbs,
                 "cache_stride": cache_stride},
    }
