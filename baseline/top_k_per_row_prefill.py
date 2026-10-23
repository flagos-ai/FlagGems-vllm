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

"""top_k_per_row_prefill baseline（DeepSeek V4 稀疏注意力 prefill 阶段逐行 top-K）。

native：torch.ops._C.top_k_per_row_prefill(logits, rowStarts, rowEnds, indices,
        numRows, stride0, stride1, topK) -> ()。indices 原地写 top-k 索引。
    Python 封装见 vllm/_custom_ops.py#L3133；STABLE_TORCH_LIBRARY 绑定签名见
    csrc/libtorch_stable/torch_bindings.cpp#L559。
    需先 import vllm._custom_ops 触发 torch.ops._C 命名空间注册。

输入构造复刻 FlagGems-vllm/benchmark/test_top_k_per_row_prefill.py 的
TopKPerRowPrefillBenchmark.get_input_iter（shape=(num_rows, vocab_size, top_k,
stride0, stride1)）：
    buf        = randn((num_rows-1)*stride0 + (vocab_size-1)*stride1 + 1) fp32
    logits     = as_strided(buf, (num_rows, vocab_size), (stride0, stride1))
    row_starts = zeros(num_rows) int32
    row_ends   = full((num_rows,), vocab_size) int32
    indices    = empty(num_rows, top_k) int32            （输出，原地写）
    调用 (logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k)
    注：部分 shape 的 stride0 > vocab_size（行间有间隔，logits 非连续），故用
    buf.as_strided(...) 忠实复刻带 stride 的内存布局（方法式调用兼容离线 stub）。

shape 来自 benchmark set_shapes：DeepSeek V4 全 vocab (64,129280,1024,129280,1)
+ DeepSeek-V4-Flash 6 档（含非连续 stride0）。
"""

import importlib

import torch

OP_NAME = "top_k_per_row_prefill"
DTYPES = [torch.float32]
IS_INPLACE = True  # indices 原地写

# benchmark set_shapes：(num_rows, vocab_size, top_k, stride0, stride1)
_SHAPES = [
    (64, 129280, 1024, 129280, 1),
    (4, 8193, 512, 8456, 1),
    (16383, 4095, 512, 4352, 1),
    (4, 16385, 512, 16648, 1),
    (12961, 4100, 512, 4360, 1),
    (16380, 5115, 512, 5376, 1),
    (4100, 1025, 512, 1288, 1),
]


def native():
    """解析 torch.ops._C.top_k_per_row_prefill；解析不到返回 None。

    先 import vllm._custom_ops 以注册 torch.ops._C 命名空间。
    """
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(getattr(torch.ops, "_C", None), "top_k_per_row_prefill", None)
    return op if callable(op) else None


def grid():
    return [
        {"num_rows": r, "vocab_size": v, "top_k": k,
         "stride0": s0, "stride1": s1}
        for (r, v, k, s0, s1) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    r = binding["num_rows"]
    v = binding["vocab_size"]
    k = binding["top_k"]
    s0 = binding["stride0"]
    s1 = binding["stride1"]
    buf = torch.randn((r - 1) * s0 + (v - 1) * s1 + 1,
                      dtype=torch.float32, device=device)
    logits = buf.as_strided((r, v), (s0, s1))
    row_starts = torch.zeros(r, dtype=torch.int32, device=device)
    row_ends = torch.full((r,), v, dtype=torch.int32, device=device)
    indices = torch.empty((r, k), dtype=torch.int32, device=device)
    return (logits, row_starts, row_ends, indices, r, s0, s1, k), {}


def key_shape(binding):
    return [binding["num_rows"], binding["vocab_size"], binding["top_k"]]


def config(binding, dtype):
    r = binding["num_rows"]
    v = binding["vocab_size"]
    k = binding["top_k"]
    return {
        "inputs": {
            "logits": {"shape": [r, v], "dtype": "torch.float32",
                       "note": "as_strided，stride=(stride0, stride1)"},
            "row_starts": {"shape": [r], "dtype": "torch.int32",
                           "note": "每行有效区间起点，全 vocab 时为 0"},
            "row_ends": {"shape": [r], "dtype": "torch.int32",
                         "note": "每行有效区间终点，全 vocab 时为 vocab_size"},
            "indices": {"shape": [r, k], "dtype": "torch.int32",
                        "note": "原地写回 top-k 索引"},
            "num_rows": {"scalar": r},
            "stride0": {"scalar": binding["stride0"]},
            "stride1": {"scalar": binding["stride1"]},
            "top_k": {"scalar": k},
        },
        "outputs": {
            "indices": {"shape": [r, k], "dtype": "torch.int32"},
        },
        "dims": {"num_rows": r, "vocab_size": v, "top_k": k,
                 "stride0": binding["stride0"], "stride1": binding["stride1"]},
    }
