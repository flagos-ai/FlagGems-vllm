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

"""top_k_per_row_decode baseline（DeepSeek V4 稀疏注意力 decode 阶段逐行 top-K）。

native：torch.ops._C.top_k_per_row_decode(logits, next_n, seq_lens, indices,
        numRows, stride0, stride1, topK) -> ()。indices 原地写 top-k 索引。
    Python 封装见 vllm/_custom_ops.py#L3155；STABLE_TORCH_LIBRARY 绑定签名见
    csrc/libtorch_stable/torch_bindings.cpp#L563。
    需先 import vllm._custom_ops 触发 torch.ops._C 命名空间注册。

输入构造复刻 FlagGems-vllm/benchmark/test_top_k_per_row_decode.py 的
TopKPerRowDecodeBenchmark.get_input_iter（shape=(num_rows, vocab_size, next_n,
top_k, stride0, stride1)）：
    buf     = randn((num_rows-1)*stride0 + (vocab_size-1)*stride1 + 1) fp32
    logits  = as_strided(buf, (num_rows, vocab_size), (stride0, stride1))
    next_n  = 1（标量）
    seq_lens = full((num_rows//next_n,), vocab_size) int32
    indices = zeros(num_rows, top_k) int32              （输出，原地写）
    调用 (logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k)
    注：benchmark 的 stride0=vocab_size、stride1=1，as_strided 结果即连续布局，
    等价 randn(num_rows, vocab_size)，故这里直接构造连续张量（兼容离线 stub）。

shape 来自 benchmark set_shapes：全部 vocab_size=262144, next_n=1, top_k=512,
stride0=262144, stride1=1，num_rows 遍历 {1,4,8,16,24,32,40,48,56,496,512}
（DeepSeek-V4-Flash decode 档位）。
"""

import importlib

import torch

OP_NAME = "top_k_per_row_decode"
DTYPES = [torch.float32]
IS_INPLACE = True  # indices 原地写

# benchmark set_shapes：(num_rows, vocab_size, next_n, top_k, stride0, stride1)
_SHAPES = [
    (1, 262144, 1, 512, 262144, 1),
    (496, 262144, 1, 512, 262144, 1),
    (512, 262144, 1, 512, 262144, 1),
    (16, 262144, 1, 512, 262144, 1),
    (32, 262144, 1, 512, 262144, 1),
    (48, 262144, 1, 512, 262144, 1),
    (40, 262144, 1, 512, 262144, 1),
    (56, 262144, 1, 512, 262144, 1),
    (4, 262144, 1, 512, 262144, 1),
    (8, 262144, 1, 512, 262144, 1),
    (24, 262144, 1, 512, 262144, 1),
]


def native():
    """解析 torch.ops._C.top_k_per_row_decode；解析不到返回 None。

    先 import vllm._custom_ops 以注册 torch.ops._C 命名空间。
    """
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(getattr(torch.ops, "_C", None), "top_k_per_row_decode", None)
    return op if callable(op) else None


def grid():
    return [
        {"num_rows": nr, "vocab_size": v, "next_n": nn,
         "top_k": k, "stride0": s0, "stride1": s1}
        for (nr, v, nn, k, s0, s1) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    nr = binding["num_rows"]
    v = binding["vocab_size"]
    nn = binding["next_n"]
    k = binding["top_k"]
    s0 = binding["stride0"]
    s1 = binding["stride1"]
    # benchmark stride0=vocab_size、stride1=1 → as_strided 即连续布局，直接构造。
    logits = torch.randn(nr, v, dtype=torch.float32, device=device)
    batch_size = nr // nn
    seq_lens = torch.full((batch_size,), v, dtype=torch.int32, device=device)
    indices = torch.zeros((nr, k), dtype=torch.int32, device=device)
    return (logits, nn, seq_lens, indices, nr, s0, s1, k), {}


def key_shape(binding):
    return [binding["num_rows"], binding["vocab_size"], binding["top_k"]]


def config(binding, dtype):
    nr = binding["num_rows"]
    v = binding["vocab_size"]
    nn = binding["next_n"]
    k = binding["top_k"]
    batch_size = nr // nn
    return {
        "inputs": {
            "logits": {"shape": [nr, v], "dtype": "torch.float32"},
            "next_n": {"scalar": nn},
            "seq_lens": {"shape": [batch_size], "dtype": "torch.int32",
                         "note": "每行有效序列长度，此处 = vocab_size"},
            "indices": {"shape": [nr, k], "dtype": "torch.int32",
                        "note": "原地写回 top-k 索引"},
            "num_rows": {"scalar": nr},
            "stride0": {"scalar": binding["stride0"]},
            "stride1": {"scalar": binding["stride1"]},
            "top_k": {"scalar": k},
        },
        "outputs": {
            "indices": {"shape": [nr, k], "dtype": "torch.int32"},
        },
        "dims": {"num_rows": nr, "next_n": nn, "vocab_size": v, "top_k": k},
    }
